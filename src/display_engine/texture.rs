use vulkanalia::prelude::v1_0::*;

#[derive(Clone, PartialEq, Debug)]
pub enum TextureColorFormat {
    GrayScale8Bit,
}

#[derive(Clone, Debug)]
pub struct TextureDescription {
    pub width: usize,
    pub height: usize,
    pub format: TextureColorFormat
}

impl PartialEq for TextureDescription {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width &&
        self.height == other.height &&
        self.format == other.format
    }
}

impl TextureDescription {
    pub fn size(&self) -> usize {
        match self.format {
            TextureColorFormat::GrayScale8Bit => self.width * self.height * 4,
            _ => 0
        }
    }

    pub fn vk_format(&self) -> vk::Format {
        match self.format {
            TextureColorFormat::GrayScale8Bit => vk::Format::B8G8R8A8_UNORM,
        }
    }
}
pub struct Texture {
    pub texture_bytes: Vec<u8>,
    pub description: TextureDescription,
}

#[derive(Debug)]
pub enum TextureError {
    FileReadError,
    DimensionsParseError,
    UnexpectedTextureDescription,
    UnexpectedFileFormat,
    UnexpectedFilePath
}


impl Texture {
    pub fn from_raw_file(path: &std::path::Path, description: &TextureDescription) -> Result<Self, TextureError> {
        if description.format != TextureColorFormat::GrayScale8Bit {
            return Err(TextureError::UnexpectedTextureDescription);
        }

        let filename = match path.file_name() {
            Some(filename) => match filename.to_str() {
                Some(name_str) => name_str,
                None => return Err(TextureError::UnexpectedFilePath)
            }
            None => return Err(TextureError::UnexpectedFilePath)
        };
        
        let pos = filename.find(".raw");
        match pos {
            None => return Err(TextureError::UnexpectedFileFormat),
            Some(_) => {}
        }

        let texture_pure_bytes = std::fs::read(path)
            .map_err(|_| TextureError::FileReadError)?;
        let mut texture_bytes = Vec::new();
        for byte in &texture_pure_bytes {
            texture_bytes.push(*byte);
            texture_bytes.push(*byte);
            texture_bytes.push(*byte);
            texture_bytes.push(255 as u8);
        }

        Ok(Texture {
            texture_bytes,
            description: description.clone(),
        })
    }

    pub fn size(&self) -> usize {
        self.texture_bytes.len()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_texture_from_raw_file() {
        let path = std::path::Path::new("data/838x1024xu8.raw");
        let desc = TextureDescription {
            width: 838,
            height: 1024,
            format: TextureColorFormat::GrayScale8Bit,
        };
        let res = Texture::from_raw_file(&path, &desc);
        assert!(res.is_ok());
        let texture = res.unwrap();
        assert_eq!(texture.description.width, 838);
        assert_eq!(texture.description.height, 1024);
        assert_eq!(texture.description.format, TextureColorFormat::GrayScale8Bit);
    }
}
