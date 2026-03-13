use crate::app::OcrApp;
use crate::data::WorkMode;

pub fn show(ctx: &egui::Context, app: &mut OcrApp) {
    egui::TopBottomPanel::bottom("status_bar")
        .exact_height(30.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Mode indicator
                ui.label(format!("Mode: {}", app.current_mode.tooltip()));
                ui.separator();
                
                // Image info
                if let Some(img) = &app.current_image {
                    ui.label(format!("{}×{}", img.width(), img.height()));
                    ui.separator();
                }
                
                // Text blocks count
                ui.label(format!("Blocks: {}", app.text_blocks.len()));
                ui.separator();
                
                // Selected count
                let selected = app.text_blocks.iter().filter(|b| b.selected).count();
                ui.label(format!("Selected: {}", selected));
                ui.separator();
                
                // Processing indicator
                if app.is_processing {
                    ui.spinner();
                    ui.label("Processing...");
                } else {
                    ui.label("✓ Ready");
                }
                
                // Batch mode progress
                if app.current_mode == WorkMode::Batch && !app.batch_files.is_empty() {
                    ui.separator();
                    ui.label(format!(
                        "Batch: {} / {}",
                        app.batch_current_index + 1,
                        app.batch_files.len()
                    ));
                    
                    if ui.button("Next →").clicked() && !app.is_processing {
                        app.process_next_batch();
                    }
                }
            });
        });
}
