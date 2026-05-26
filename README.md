# Hexapod Fiksatör — Web Uygulaması

Taylor Spatial Frame tasarım ve düzeltme programı hesap sistemi.

## Railway Deployment

1. GitHub'a push et
2. Railway → New Project → Deploy from GitHub Repo
3. Otomatik deploy — PORT ortam değişkeni Railway tarafından set edilir

## Lokal Çalıştırma

```bash
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

## API Endpoints

| Endpoint | Method | Açıklama |
|---|---|---|
| `/api/ring` | GET | Halka geometrisi |
| `/api/all-rings` | GET | Tüm halka boyutları |
| `/api/correction` | POST | Düzeltme programı hesabı |
| `/api/export/csv` | POST | Takvim CSV |
| `/api/export/coords` | GET | Koordinat CSV |

## Özellikler

- 2D teknik çizim (SVG)
- 3D montaj görünümü (Three.js)
- Tüm halkalar karşılaştırma
- Koordinat tablosu
- Düzeltme programı (inverse kinematik)
- SVG + CSV export
