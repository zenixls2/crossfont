//! Rasterization powered by DirectWrite.

use std::borrow::Cow;
use std::collections::HashMap;
use std::ffi::OsString;
use std::os::windows::ffi::OsStringExt;
// use std::os::windows::ffi::{OsStrExt, OsStringExt};

use crate::{Info, RasterizeExt};

use dwrote::{
    FontCollection, FontFace, FontFallback, FontStretch, FontStyle, FontWeight, GlyphOffset,
    GlyphRunAnalysis, TextAnalysisSource, TextAnalysisSourceMethods, DWRITE_GLYPH_RUN,
};

use winapi::shared::ntdef::{HRESULT, LOCALE_NAME_MAX_LENGTH};
use winapi::shared::winerror::*;
use winapi::um::dwrite;
use winapi::um::unknwnbase::IUnknown;
use winapi::um::winnls::GetUserDefaultLocaleName;
use winapi::Interface;
use wio::com::ComPtr;

use super::{
    BitmapBuffer, Error, FontDesc, FontKey, GlyphKey, KeyType, Metrics, RasterizedGlyph, Size,
    Slant, Style, Weight,
};

/// DirectWrite uses 0 for missing glyph symbols.
/// https://docs.microsoft.com/en-us/typography/opentype/spec/recom#glyph-0-the-notdef-glyph
const MISSING_GLYPH_INDEX: u16 = 0;

/// Cached DirectWrite font.
struct Font {
    face: FontFace,
    family_name: String,
    weight: FontWeight,
    style: FontStyle,
    stretch: FontStretch,
}

pub struct DirectWriteRasterizer {
    fonts: HashMap<FontKey, Font>,
    keys: HashMap<FontDesc, FontKey>,
    device_pixel_ratio: f32,
    available_fonts: FontCollection,
    fallback_sequence: Option<FontFallback>,
    analyzer: ComPtr<dwrite::IDWriteTextAnalyzer>,
    locale: Vec<u16>,
}

impl DirectWriteRasterizer {
    fn rasterize_glyph(
        &self,
        face: &FontFace,
        size: Size,
        glyph_index: u16,
    ) -> Result<RasterizedGlyph, Error> {
        let em_size = em_size(size);

        let glyph_run = DWRITE_GLYPH_RUN {
            fontFace: unsafe { face.as_ptr() },
            fontEmSize: em_size,
            glyphCount: 1,
            glyphIndices: &glyph_index,
            glyphAdvances: &0.0,
            glyphOffsets: &GlyphOffset::default(),
            isSideways: 0,
            bidiLevel: 0,
        };

        let rendering_mode = face.get_recommended_rendering_mode_default_params(
            em_size,
            self.device_pixel_ratio,
            dwrote::DWRITE_MEASURING_MODE_NATURAL,
        );

        let glyph_analysis = GlyphRunAnalysis::create(
            &glyph_run,
            self.device_pixel_ratio,
            None,
            rendering_mode,
            dwrote::DWRITE_MEASURING_MODE_NATURAL,
            0.0,
            0.0,
        )?;

        let bounds =
            glyph_analysis.get_alpha_texture_bounds(dwrote::DWRITE_TEXTURE_CLEARTYPE_3x1)?;

        let buffer = BitmapBuffer::RGB(
            glyph_analysis.create_alpha_texture(dwrote::DWRITE_TEXTURE_CLEARTYPE_3x1, bounds)?,
        );

        Ok(RasterizedGlyph {
            character: KeyType::GlyphIndex(glyph_index.into()),
            width: (bounds.right - bounds.left) as i32,
            height: (bounds.bottom - bounds.top) as i32,
            top: -bounds.top,
            left: bounds.left,
            buffer,
        })
    }

    fn get_loaded_font(&self, font_key: FontKey) -> Result<&Font, Error> {
        self.fonts.get(&font_key).ok_or(Error::UnknownFontKey)
    }

    #[inline]
    fn get_char_index(&self, face: &FontFace, c: char) -> u16 {
        face.get_glyph_indices(&[c as u32])
            .first()
            // DirectWrite returns 0 if the glyph does not exist in the font.
            .filter(|glyph_index| **glyph_index != 0)
            .copied()
            .unwrap_or(MISSING_GLYPH_INDEX)
    }

    fn get_fallback_font(&self, loaded_font: &Font, character: char) -> Option<dwrote::Font> {
        let fallback = self.fallback_sequence.as_ref()?;

        let mut buffer = [0u16; 2];
        character.encode_utf16(&mut buffer);

        let length = character.len_utf16() as u32;
        let utf16_codepoints = &buffer[..length as usize];

        let locale = get_current_locale();

        let text_analysis_source_data = TextAnalysisSourceData { locale: &locale, length };
        let text_analysis_source = TextAnalysisSource::from_text(
            Box::new(text_analysis_source_data),
            Cow::Borrowed(utf16_codepoints),
        );

        let fallback_result = fallback.map_characters(
            &text_analysis_source,
            0,
            length,
            &self.available_fonts,
            Some(&loaded_font.family_name),
            loaded_font.weight,
            loaded_font.style,
            loaded_font.stretch,
        );

        fallback_result.mapped_font
    }
}

impl crate::Rasterize for DirectWriteRasterizer {
    fn new(
        device_pixel_ratio: f32,
        _: bool,
        _ligatures: bool,
    ) -> Result<DirectWriteRasterizer, Error> {
        let analyzer = unsafe {
            let mut factory: *mut dwrite::IDWriteFactory = std::ptr::null_mut();
            let hr = dwrite::DWriteCreateFactory(
                dwrite::DWRITE_FACTORY_TYPE_SHARED,
                &dwrite::IDWriteFactory::uuidof(),
                &mut factory as *mut *mut dwrite::IDWriteFactory as *mut *mut IUnknown,
            );
            assert_eq!(hr, 0, "error creating dwrite factory");
            let mut native: *mut dwrite::IDWriteTextAnalyzer = std::ptr::null_mut();
            let hr = (*factory).CreateTextAnalyzer(&mut native);
            assert_eq!(hr, 0, "IDWriteTextAnalyzer init fail");
            factory.as_ref().map(|x| x.Release());
            ComPtr::from_raw(native)
        };
        let mut locale = vec![0u16; LOCALE_NAME_MAX_LENGTH];
        let _locale_len =
            unsafe { GetUserDefaultLocaleName(locale.as_mut_ptr(), locale.len() as i32) };
        Ok(DirectWriteRasterizer {
            fonts: HashMap::new(),
            keys: HashMap::new(),
            device_pixel_ratio,
            available_fonts: FontCollection::system(),
            fallback_sequence: FontFallback::get_system_fallback(),
            analyzer,
            locale,
        })
    }

    fn metrics(&self, key: FontKey, size: Size) -> Result<Metrics, Error> {
        let face = &self.get_loaded_font(key)?.face;
        let vmetrics = face.metrics().metrics0();

        let scale = em_size(size) * self.device_pixel_ratio / f32::from(vmetrics.designUnitsPerEm);

        let underline_position = f32::from(vmetrics.underlinePosition) * scale;
        let underline_thickness = f32::from(vmetrics.underlineThickness) * scale;

        let strikeout_position = f32::from(vmetrics.strikethroughPosition) * scale;
        let strikeout_thickness = f32::from(vmetrics.strikethroughThickness) * scale;

        let ascent = f32::from(vmetrics.ascent) * scale;
        let descent = -f32::from(vmetrics.descent) * scale;
        let line_gap = f32::from(vmetrics.lineGap) * scale;

        let line_height = f64::from(ascent - descent + line_gap);

        // Since all monospace characters have the same width, we use `!` for horizontal metrics.
        let character = '!';
        let glyph_index = self.get_char_index(face, character);

        let glyph_metrics = face.get_design_glyph_metrics(&[glyph_index], false);
        let hmetrics = glyph_metrics.first().ok_or(Error::MetricsNotFound)?;

        let average_advance = f64::from(hmetrics.advanceWidth) * f64::from(scale);

        Ok(Metrics {
            descent,
            average_advance,
            line_height,
            underline_position,
            underline_thickness,
            strikeout_position,
            strikeout_thickness,
        })
    }

    fn load_font(&mut self, desc: &FontDesc, _size: Size) -> Result<FontKey, Error> {
        // Fast path if face is already loaded.
        if let Some(key) = self.keys.get(desc) {
            return Ok(*key);
        }

        let family = self
            .available_fonts
            .get_font_family_by_name(&desc.name)
            .ok_or_else(|| Error::FontNotFound(desc.clone()))?;

        let font = match desc.style {
            Style::Description { weight, slant } => {
                // This searches for the "best" font - should mean we don't have to worry about
                // fallbacks if our exact desired weight/style isn't available.
                Ok(family.get_first_matching_font(weight.into(), FontStretch::Normal, slant.into()))
            },
            Style::Specific(ref style) => {
                let mut idx = 0;
                let count = family.get_font_count();

                loop {
                    if idx == count {
                        break Err(Error::FontNotFound(desc.clone()));
                    }

                    let font = family.get_font(idx);

                    if font.face_name() == *style {
                        break Ok(font);
                    }

                    idx += 1;
                }
            },
        }?;

        let key = FontKey::next();
        self.keys.insert(desc.clone(), key);
        self.fonts.insert(key, font.into());

        Ok(key)
    }

    fn get_glyph(&mut self, glyph: GlyphKey) -> Result<RasterizedGlyph, Error> {
        let loaded_font = self.get_loaded_font(glyph.font_key)?;

        let loaded_fallback_font;
        let mut font = loaded_font;
        let glyph_index = match glyph.id {
            KeyType::Char(character) => {
                let mut index = self.get_char_index(&loaded_font.face, character);
                if index == MISSING_GLYPH_INDEX {
                    if let Some(fallback_font) = self.get_fallback_font(&loaded_font, character) {
                        loaded_fallback_font = Font::from(fallback_font);
                        font = &loaded_fallback_font;
                        index = self.get_char_index(&loaded_fallback_font.face, character);
                    }
                }
                index
            },
            KeyType::GlyphIndex(index) => index as u16,
            KeyType::Placeholder => {
                return Ok(RasterizedGlyph {
                    character: KeyType::Placeholder,
                    width: 0,
                    height: 0,
                    top: 0,
                    left: 0,
                    buffer: BitmapBuffer::RGB(Vec::new()),
                });
            },
        };

        let rasterized_glyph = self.rasterize_glyph(&font.face, glyph.size, glyph_index)?;

        if glyph_index == MISSING_GLYPH_INDEX {
            Err(Error::MissingGlyph(rasterized_glyph))
        } else {
            Ok(rasterized_glyph)
        }
    }

    fn update_dpr(&mut self, device_pixel_ratio: f32) {
        self.device_pixel_ratio = device_pixel_ratio;
    }
}

impl RasterizeExt for DirectWriteRasterizer {
    fn shape(&mut self, text: &str, font_key: FontKey) -> Vec<Info> {
        let face = &self.get_loaded_font(font_key).unwrap().face;
        unsafe {
            let string: Vec<u16> = text.encode_utf16().collect();
            let max_glyphs = 3 * string.len() as u32 / 2 + 16;
            let mut cluster_map = vec![0u16; string.len()];
            let mut text_props = vec![dwrite::DWRITE_SHAPING_TEXT_PROPERTIES { bit_fields: 0 }];
            let mut glyph_indices = vec![0u16; max_glyphs as usize];
            let mut glyph_props = vec![dwrite::DWRITE_SHAPING_GLYPH_PROPERTIES { bit_fields: 0 }];

            let mut analysis = dwrite::DWRITE_SCRIPT_ANALYSIS {
                script: 0,
                shapes: dwrite::DWRITE_SCRIPT_SHAPES_DEFAULT,
            };
            let liga = dwrite::DWRITE_FONT_FEATURE {
                nameTag: dwrite::DWRITE_FONT_FEATURE_TAG_STANDARD_LIGATURES,
                parameter: 1,
            };
            let calt = dwrite::DWRITE_FONT_FEATURE {
                nameTag: dwrite::DWRITE_FONT_FEATURE_TAG_CONTEXTUAL_ALTERNATES,
                parameter: 1,
            };
            let mut features_unit = dwrite::DWRITE_TYPOGRAPHIC_FEATURES {
                features: [liga, calt].as_mut_ptr(),
                featureCount: 2,
            };
            let features =
                [&mut features_unit as *const dwrite::DWRITE_TYPOGRAPHIC_FEATURES].as_mut_ptr();
            let mut glyph_count = 0u32;
            let hr = (*self.analyzer).GetGlyphs(
                string.as_ptr(),
                string.len() as u32,
                face.as_ptr(),
                false as winapi::ctypes::c_int,
                false as winapi::ctypes::c_int,
                &mut analysis as *const _ as *mut _,
                self.locale.as_ptr(),
                std::ptr::null_mut(), // self.substitution.as_mut_ptr(),
                features,
                [string.len() as u32].as_ptr(),
                1,
                max_glyphs,
                cluster_map.as_mut_ptr(),
                text_props.as_mut_ptr(),
                glyph_indices.as_mut_ptr(),
                glyph_props.as_mut_ptr(),
                &mut glyph_count as *mut _,
            );
            if hr == HRESULT_FROM_WIN32(ERROR_INVALID_PARAMETER) {
                panic!("invalid parameter");
            }
            assert_eq!(hr, 0, "error get glyphs");
            assert!(
                glyph_count <= string.len() as u32,
                "glyph count more then string len is currently unsupported"
            );
            // not pretty sure about the mapping of cluster.
            // and if codepoint won't be zero, there's no meaning to do such conversion.
            let mut state = -1;
            let clusters: Vec<u32> = cluster_map
                .iter()
                .enumerate()
                .filter_map(move |(i, shift)| {
                    if state == *shift as i32 {
                        None
                    } else {
                        state = *shift as i32;
                        Some(i as u32)
                    }
                })
                .collect();
            glyph_indices
                .iter()
                .take(glyph_count as usize)
                .enumerate()
                .map(|(i, codepoint)| Info { codepoint: *codepoint as u32, cluster: clusters[i] })
                .collect()
        }
    }
}

fn em_size(size: Size) -> f32 {
    size.as_f32_pts() * (96.0 / 72.0)
}

impl From<dwrote::Font> for Font {
    fn from(font: dwrote::Font) -> Font {
        Font {
            face: font.create_font_face(),
            family_name: font.family_name(),
            weight: font.weight(),
            style: font.style(),
            stretch: font.stretch(),
        }
    }
}

impl From<Weight> for FontWeight {
    fn from(weight: Weight) -> FontWeight {
        match weight {
            Weight::Bold => FontWeight::Bold,
            Weight::Normal => FontWeight::Regular,
        }
    }
}

impl From<Slant> for FontStyle {
    fn from(slant: Slant) -> FontStyle {
        match slant {
            Slant::Oblique => FontStyle::Oblique,
            Slant::Italic => FontStyle::Italic,
            Slant::Normal => FontStyle::Normal,
        }
    }
}

fn get_current_locale() -> String {
    let mut buffer = vec![0u16; LOCALE_NAME_MAX_LENGTH];
    let len =
        unsafe { GetUserDefaultLocaleName(buffer.as_mut_ptr(), buffer.len() as i32) as usize };

    // `len` includes null byte, which we don't need in Rust.
    OsString::from_wide(&buffer[..len - 1]).into_string().expect("Locale not valid unicode")
}

/// Font fallback information for dwrote's TextAnalysisSource.
struct TextAnalysisSourceData<'a> {
    locale: &'a str,
    length: u32,
}

impl TextAnalysisSourceMethods for TextAnalysisSourceData<'_> {
    fn get_locale_name(&self, _text_position: u32) -> (Cow<str>, u32) {
        (Cow::Borrowed(self.locale), self.length)
    }

    fn get_paragraph_reading_direction(&self) -> dwrite::DWRITE_READING_DIRECTION {
        dwrite::DWRITE_READING_DIRECTION_LEFT_TO_RIGHT
    }
}

impl From<HRESULT> for Error {
    fn from(hresult: HRESULT) -> Self {
        let message = format!("a DirectWrite rendering error occurred: {:X}", hresult);
        Error::PlatformError(message)
    }
}
mod tests {
    #[test]
    fn build_font_and_get_glyph() {
        use super::*;
        use crate::Rasterize;
        let mut rasterizer = DirectWriteRasterizer::new(6., false, true).unwrap();
        let font = rasterizer
            .load_font(
                &FontDesc {
                    name: "Consolas".to_string(),
                    style: Style::Description { slant: Slant::Normal, weight: Weight::Normal },
                },
                Size::new(12.),
            )
            .unwrap();
        for c in &['a', 'b', '!', '日'] {
            let key = GlyphKey { id: KeyType::Char(*c), font_key: font, size: Size::new(12.) };
            let glyph = rasterizer.get_glyph(key).unwrap();
            let buf = match &glyph.buffer {
                BitmapBuffer::RGB(buf) => buf,
                BitmapBuffer::RGBA(buf) => buf,
            };

            // Debug the glyph
            for row in 0..glyph.height {
                for col in 0..glyph.width {
                    let index = ((glyph.width * 3 * row) + (col * 3)) as usize;
                    let value = buf[index];
                    let c = match value {
                        0..=50 => ' ',
                        51..=100 => '.',
                        101..=150 => '~',
                        151..=200 => '*',
                        201..=255 => '#',
                    };
                    print!("{}", c);
                }
                println!();
            }
        }
    }
    #[test]
    fn shape() {
        use super::*;
        use crate::Rasterize;
        let mut r = DirectWriteRasterizer::new(1.0, false, true).unwrap();
        let font_desc = FontDesc::new("Consolas", Style::Specific("Regular".to_string()));
        let font_key = r.load_font(&font_desc, Size(16)).unwrap();

        let face = &r.get_loaded_font(font_key).unwrap().face;
        let v: Vec<u16> =
            ['-', '-', '>', '<', '-'].iter().map(|c| r.get_char_index(&face, *c)).collect();
        println!("{:?}", v);

        let infos = r.shape("--><-", font_key);
        println!("{:?}", infos);

        let mut key = GlyphKey { id: 0.into(), font_key, size: Size(16) };
        for (info, index) in infos.into_iter().zip(v.into_iter()) {
            if info.codepoint != 0 {
                assert_eq!(info.codepoint, index as u32);
                key.id = info.codepoint.into();
                r.get_glyph(key).unwrap();
            }
        }
    }
}
