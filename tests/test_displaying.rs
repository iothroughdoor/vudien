use multi_platform_medical_imaging::display_engine::graphics_pipeline::Vertex;
use multi_platform_medical_imaging::display_engine::DisplayEngine;
use multi_platform_medical_imaging::display_engine::DisplayEngineError;
use cgmath::{vec2, vec3};

static VERTICES: [Vertex; 4] = [
    Vertex{position: vec2(-0.5, -0.5), color: vec3(u8::MAX, 0, 0)            , _padding: 0, tex_coord: vec2(0.0, 0.0)},
    Vertex{position: vec2(0.5, -0.5) , color: vec3(0, u8::MAX, 0)            , _padding: 0, tex_coord: vec2(1.0, 0.0)},
    Vertex{position: vec2(0.5, 0.5)  , color: vec3(0, 0, u8::MAX)            , _padding: 0, tex_coord: vec2(1.0, 1.0)},
    Vertex{position: vec2(-0.5, 0.5) , color: vec3(u8::MAX, u8::MAX, u8::MAX), _padding: 0, tex_coord: vec2(0.0, 1.0)},
];

const INDICES: &[u16] = &[0, 1, 2, 2, 3, 0];

fn main() -> Result<(), DisplayEngineError> {
    const WIDTH: i32 = 976;
    const HEIGHT: i32 = 976;

    let event_loop = winit::event_loop::EventLoop::new().unwrap();

    let window = winit::window::WindowBuilder::new()
        .with_title("View A")
        .with_inner_size(winit::dpi::LogicalSize::new(WIDTH, HEIGHT))
        .build(&event_loop)
        .unwrap();

    let texture = multi_platform_medical_imaging::display_engine::texture::Texture::from_raw_file(
        &std::path::Path::new("data/976x976xu8.raw")
    ).unwrap();

    let mut display_engine = DisplayEngine::new(&window)?;
    display_engine.upload_vertices(&VERTICES, &INDICES)?;
    display_engine.upload_texture(texture)?;
    event_loop.run(move |event, elwt| {
        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => {
                    display_engine.wait();
                    elwt.exit();
                }
                winit::event::WindowEvent::RedrawRequested => {
                    display_engine.display().unwrap();
                }
                _ => {}
            },
            _ => {}
        }
    }).unwrap();
    println!("This is the end of the line.");

    Ok(())
}
