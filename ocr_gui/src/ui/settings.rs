use crate::app::OcrApp;
use crate::config::{OcrConfig, Theme};

pub fn show(ctx: &egui::Context, visible: &mut bool, config: &mut OcrConfig, app: &mut OcrApp) {
    egui::Window::new("⚙ Settings")
        .open(visible)
        .collapsible(false)
        .resizable(false)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Appearance");
                ui.separator();
                
                ui.horizontal(|ui| {
                    ui.label("Theme:");
                    if ui.selectable_label(config.theme == Theme::Dark, "🌙 Dark").clicked() {
                        config.theme = Theme::Dark;
                    }
                    if ui.selectable_label(config.theme == Theme::Light, "☀ Light").clicked() {
                        config.theme = Theme::Light;
                    }
                });
                
                ui.add_space(10.0);
                ui.heading("OCR Settings");
                ui.separator();
                
                ui.horizontal(|ui| {
                    ui.label("Min Confidence:");
                    ui.add(egui::Slider::new(&mut config.min_confidence, 0.0..=1.0).step_by(0.05));
                });
                
                ui.horizontal(|ui| {
                    ui.label("Backend:");
                    ui.label(&config.backend);
                    ui.label("(auto-detected)");
                });
                
                ui.add_space(10.0);
                ui.heading("Model Paths");
                ui.separator();
                
                ui.label("Detection Model:");
                ui.text_edit_singleline(&mut config.det_model_path);
                
                ui.label("Recognition Model:");
                ui.text_edit_singleline(&mut config.rec_model_path);
                
                ui.label("Charset:");
                ui.text_edit_singleline(&mut config.charset_path);
                
                ui.add_space(10.0);
                ui.heading("Data");
                ui.separator();
                
                ui.horizontal(|ui| {
                    ui.label("History Directory:");
                    ui.text_edit_singleline(&mut config.history_dir);
                });
                
                ui.add_space(10.0);
                ui.separator();
                
                ui.horizontal(|ui| {
                    if ui.button("💾 Save").clicked() {
                        if let Err(e) = config.save() {
                            tracing::error!("Failed to save config: {}", e);
                        }
                        
                        // Reinitialize engine with new settings
                        if let Err(e) = app.init_engine() {
                            app.error_message = Some(format!("Failed to reinitialize engine: {}", e));
                        }
                    }
                    
                    if ui.button("🔄 Reset to Defaults").clicked() {
                        *config = OcrConfig::default();
                    }
                    
                    if ui.button("❌ Close").clicked() {
                        *visible = false;
                    }
                });
            });
        });
}
