use crate::app::OcrApp;
use crate::data::WorkMode;

pub fn show(ui: &mut egui::Ui, app: &mut OcrApp) {
    ui.vertical_centered(|ui| {
        ui.add_space(10.0);
        
        // Mode buttons
        for mode in [WorkMode::File, WorkMode::Batch, WorkMode::History] {
            let is_active = app.current_mode == mode;
            
            let button = egui::Button::new(
                egui::RichText::new(mode.icon()).size(24.0)
            )
            .fill(if is_active {
                egui::Color32::from_rgb(60, 60, 200)
            } else {
                egui::Color32::TRANSPARENT
            })
            .min_size(egui::vec2(50.0, 50.0));
            
            if ui.add(button).on_hover_text(mode.tooltip()).clicked() {
                app.current_mode = mode;
                
                // Open file dialog when switching to file or batch mode
                match mode {
                    WorkMode::File => app.open_file_dialog(),
                    WorkMode::Batch => app.open_batch_dialog(),
                    WorkMode::History => {}
                }
            }
            
            ui.add_space(5.0);
        }
        
        // Push settings button to bottom
        ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
            ui.add_space(10.0);
            
            let settings_button = egui::Button::new(
                egui::RichText::new("⚙").size(24.0)
            )
            .fill(if app.settings_visible {
                egui::Color32::from_rgb(60, 60, 200)
            } else {
                egui::Color32::TRANSPARENT
            })
            .min_size(egui::vec2(50.0, 50.0));
            
            if ui.add(settings_button).on_hover_text("Settings").clicked() {
                app.settings_visible = !app.settings_visible;
            }
        });
    });
}
