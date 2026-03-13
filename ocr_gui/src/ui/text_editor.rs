use crate::app::OcrApp;

pub fn show(ui: &mut egui::Ui, app: &mut OcrApp) {
    ui.vertical(|ui| {
        ui.heading("OCR Results");
        ui.separator();
        
        // Show selected count and actions
        let selected_count = app.text_blocks.iter().filter(|b| b.selected).count();
        ui.horizontal(|ui| {
            ui.label(format!("Selected: {} / {}", selected_count, app.text_blocks.len()));
            
            if ui.button("Select All").clicked() {
                for block in &mut app.text_blocks {
                    block.selected = true;
                }
            }
            
            if ui.button("Clear").clicked() {
                for block in &mut app.text_blocks {
                    block.selected = false;
                }
            }
        });
        
        ui.separator();
        
        // Show text blocks
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for (idx, block) in app.text_blocks.iter_mut().enumerate() {
                    if !block.selected {
                        continue;
                    }
                    
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(format!("#{}", idx + 1));
                            ui.label(format!("Confidence: {:.1}%", block.confidence * 100.0));
                        });
                        
                        ui.add(
                            egui::TextEdit::multiline(&mut block.text)
                                .desired_width(f32::INFINITY)
                                .desired_rows(2)
                        );
                    });
                    
                    ui.add_space(5.0);
                }
                
                if selected_count == 0 {
                    ui.centered_and_justified(|ui| {
                        ui.label("No text blocks selected.\nClick on text boxes in the image to select them.");
                    });
                }
            });
        
        ui.separator();
        
        // Export buttons
        ui.horizontal(|ui| {
            if ui.button("📋 Copy").clicked() && selected_count > 0 {
                app.copy_selected_text();
            }
            
            if ui.button("💾 Export Text").clicked() && selected_count > 0 {
                app.export_text();
            }
            
            if ui.button("📄 Export JSON").clicked() && selected_count > 0 {
                app.export_json();
            }
        });
    });
}
