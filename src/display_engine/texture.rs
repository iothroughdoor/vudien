
pub struct TextureFormat {
    pub width: usize,
    pub height: usize,
    pub bytes_per_pixel: usize,
}

pub struct Texture {
    pub texture_bytes: Vec<u8>,
    pub format: TextureFormat,
}

#[derive(Debug)]
pub enum TextureError {
    FileReadError,
    DimensionsParseError,
    UnexpectedFileFormat,
    UnexpectedFilePath
}

impl PartialEq for TextureFormat {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width &&
        self.height == other.height &&
        self.bytes_per_pixel == other.bytes_per_pixel
    }
}

impl Texture {
    pub fn from_raw_file(path: &std::path::Path) -> Result<Self, TextureError> {
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
        let stripped_path = filename.replace(".raw", "");
        let name_fields = stripped_path.split("x")
                                       .collect::<Vec<&str>>();
        let width = name_fields[0].parse::<u32>().map_err(|_| TextureError::DimensionsParseError)?;
        let height = name_fields[1].parse::<u32>().map_err(|_| TextureError::DimensionsParseError)?;
        if name_fields[2] != "u8" {
            return Err(TextureError::UnexpectedFileFormat);
        }
        let bytes_per_pixel = 4;

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
            format: TextureFormat {
                width: width as usize,
                height: height as usize,
                bytes_per_pixel
            }
        })
    }

    pub fn size(&self) -> usize {
        self.texture_bytes.len()
    }
}


mod tests {
    use super::*;

    #[test]
    fn test_texture_from_raw_file() {
        let path = std::path::Path::new("data/976x976xu8.raw");
        let res = Texture::from_raw_file(&path);
        assert!(res.is_ok());
        let texture = res.unwrap();
        assert_eq!(texture.format.width, 976);
        assert_eq!(texture.format.height, 976);
        assert_eq!(texture.format.bytes_per_pixel, 4);
    }
}