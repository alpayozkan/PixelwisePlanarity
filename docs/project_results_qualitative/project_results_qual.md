# Qualitative Results

For each sample: **Row 1** shows plane segmentation (RGB | GT | predictions), **Row 2** shows inlier (green) / outlier (red) overlays with precision and recall.

Methods: MoGe (Ours), ZeroPlane (finetuned, dinov2, 60k), ZeroPlane (released), MoGe (Ours Indoors), ZeroPlane (finetuned Indoors, dinov2, 60k)

## ScanNet++ (indoor real)

Inlier threshold: 0.5cm

![000_bde1e479ad_frame_005650](scannetpp/samples/000_bde1e479ad_frame_005650.png)

![001_5ee7c22ba0_frame_001525](scannetpp/samples/001_5ee7c22ba0_frame_001525.png)

![002_1ada7a0617_frame_004825](scannetpp/samples/002_1ada7a0617_frame_004825.png)

## Hypersim (indoor synthetic)

Inlier threshold: 0.5cm

![000_ai_055_001_0030](hypersim/samples/000_ai_055_001_0030.png)

![001_ai_008_006_0073](hypersim/samples/001_ai_008_006_0073.png)

![002_ai_002_007_0009](hypersim/samples/002_ai_002_007_0009.png)

## Synthia (outdoor synthetic)

Inlier threshold: 5.0cm

![000_test5_16segs_weather_0_spawn_0_roadTexture_1_P_None_C_None_B_None_WC_None_000146](synthia/samples/000_test5_16segs_weather_0_spawn_0_roadTexture_1_P_None_C_None_B_None_WC_None_000146.png)

![001_test5_11segs_weather_2_spawn_2_roadTexture_0_P_None_C_None_B_None_WC_None_000346](synthia/samples/001_test5_11segs_weather_2_spawn_2_roadTexture_0_P_None_C_None_B_None_WC_None_000346.png)

![002_test5_22segs_weather_4_spawn_1_roadTexture_1_P_None_C_None_B_None_WC_None_000456](synthia/samples/002_test5_22segs_weather_4_spawn_1_roadTexture_1_P_None_C_None_B_None_WC_None_000456.png)

## VKITTI2 (outdoor synthetic)

Inlier threshold: 5.0cm

![000_Scene18_morning_00240](vkitti2/samples/000_Scene18_morning_00240.png)

![001_Scene18_15-deg-right_00170](vkitti2/samples/001_Scene18_15-deg-right_00170.png)

![002_Scene20_30-deg-left_00550](vkitti2/samples/002_Scene20_30-deg-left_00550.png)
