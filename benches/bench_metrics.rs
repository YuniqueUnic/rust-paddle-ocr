use criterion::{criterion_group, criterion_main, Criterion};
use image::DynamicImage;
use ocr_rs::{DetModel, DetOptions, DetPrecisionMode, RecModel, RecOptions};
use std::time::Duration;

fn setup() -> (DetModel, RecModel, DynamicImage) {
    // 加载模型 - 在性能测试前完成
    let det = DetModel::from_file("./models/ch_PP-OCRv4_det_infer.mnn", None)
        .expect("Failed to load detection model")
        .with_options(
            DetOptions::new()
                .with_box_border(12)
                .with_merge_boxes(false)
                .with_merge_threshold(1)
                .with_precision_mode(DetPrecisionMode::Fast),
        );

    let rec = RecModel::from_file(
        "./models/ch_PP-OCRv4_rec_infer.mnn",
        "./models/ppocr_keys_v4.txt",
        None,
    )
    .expect("Failed to load recognition model")
    .with_options(RecOptions::new());

    // 加载测试图片 - 在性能测试前完成
    let img = image::open("./res/1.png").expect("Failed to load test image");

    (det, rec, img)
}

fn bench_detection(c: &mut Criterion) {
    let (det, _, img) = setup();

    let mut group = c.benchmark_group("text_detection");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("det_model", |b| {
        b.iter(|| {
            det.detect(&img).expect("Detection failed");
        });
    });

    group.finish();
}

fn bench_recognition(c: &mut Criterion) {
    let (det, rec, img) = setup();

    // 先检测文本区域并裁剪
    let detections = det
        .detect_and_crop(&img)
        .expect("Failed to detect and crop text images");

    let mut group = c.benchmark_group("text_recognition");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("rec_model", |b| {
        b.iter(|| {
            // 仅测试第一个文本区域的识别，如果没有文本区域则跳过
            if let Some((text_img, _)) = detections.first() {
                rec.recognize(text_img).expect("Recognition failed");
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench_detection, bench_recognition,);
criterion_main!(benches);
