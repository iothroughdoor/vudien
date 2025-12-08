use multi_platform_medical_imaging::display_engine::DisplayEngine;
use multi_platform_medical_imaging::display_engine::DisplayEngineError;
use multi_platform_medical_imaging::display_engine::texture::{Texture, TextureColorFormat, TextureDescription};

fn main() -> Result<(), DisplayEngineError> {
    const WIDTH: i32 =  838;  
    const HEIGHT: i32 = 1024;

    let event_loop = winit::event_loop::EventLoop::new().unwrap();

    let window = winit::window::WindowBuilder::new()
        .with_title("View A")
        .with_inner_size(winit::dpi::LogicalSize::new(WIDTH, HEIGHT))
        .build(&event_loop)
        .unwrap();


    let description = TextureDescription {
        width: WIDTH as usize,
        height: HEIGHT as usize,
        format: TextureColorFormat::GrayScale8Bit,
    };
    let texture = Texture::from_raw_file(&std::path::Path::new("data/838x1024xu8.raw"), &description).unwrap();
    let mut display_engine = DisplayEngine::new(&window, &description)?;
    display_engine.upload_texture(texture)?;
    event_loop.run(move |event, elwt| {
        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => {
                    display_engine.wait();
                    elwt.exit();
                }
                winit::event::WindowEvent::RedrawRequested => {
                    display_engine.display(&window).unwrap();
                }
                _ => {}
            },
            _ => {}
        }
    }).unwrap();

    Ok(())
}
