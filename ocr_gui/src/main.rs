mod app;
mod config;
mod data;
mod theme;
mod ui;

use app::OcrApp;
use tracing_subscriber;

fn main() -> Result<(), eframe::Error> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Configure native options
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 600.0])
            .with_title("OCR GUI"),
        ..Default::default()
    };

    // Run the application
    eframe::run_native(
        "OCR GUI",
        native_options,
        Box::new(|cc| Ok(Box::new(OcrApp::new(cc)))),
    )
}
