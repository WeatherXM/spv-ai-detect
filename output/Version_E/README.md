# WeatherXM Video Assets — Version E

Generated in accordance with website video specifications (`docs/video.md`).

## Treatment Description
- **Version E**: Station fixed match-cut (7.5 photos/sec pace @ 15fps)
- **Pacing**: 2 frames/photo @ 15 fps

## Asset Bundle Structure
All required website video assets are placed directly in this folder for direct copy/pasting into `weatherxmcom-website/public/assets/videos/`:

| Asset File | Target Specs | Codec & Container | Purpose |
| :--- | :--- | :--- | :--- |
| `station_matchcut_720p.mp4` | 1280×720, ~1.0 Mbps | H.264 High Profile (+faststart, -an, BT.709) | Universal desktop background video |
| `station_matchcut_720p.webm` | 1280×720, ~900 kbps | VP9 (-an, Closed GOP, BT.709) | Modern high-efficiency desktop browsers |
| `station_matchcut_mobile.mp4` | 854×480, ~500 kbps | H.264 Baseline (+faststart, -an, BT.709) | Universal mobile stream (iOS Safari / Android) |
| `station_matchcut_mobile.webm` | 854×480, ~450 kbps | VP9 (-an, Closed GOP, BT.709) | Cellular bandwidth-saver stream |
| `station_matchcut-poster.webp` | 1280×720, ~60 KB | WebP (Q85, Frame 0) | Instant first-paint & Data-Saver fallback |
| `network-matchcut-poster.webp` | 1280×720, ~60 KB | WebP (Q85, Frame 0) | Alias for network.astro poster attribute |

## One-Click Deployment to Website
To deploy these assets directly into `weatherxmcom-website`:
```bash
./DEPLOY_TO_WEBSITE.sh
```
Or run manually from terminal:
```bash
cp station_matchcut* network-matchcut* path/to/weatherxmcom-website/public/assets/videos/
```

## Frontend Integration
See `embed_snippet.html` in this folder for ready-to-paste Astro `<StationVideoMesh>` and HTML5 `<video>` elements.
