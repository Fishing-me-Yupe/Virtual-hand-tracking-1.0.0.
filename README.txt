3D Hand Tracker の起動方法
1) このフォルダで:  python3 -m http.server 8000
2) Chrome / Edge で http://localhost:8000/ を開く  (カメラは localhost か https でのみ使えます)
3) 「カメラを開始」。カメラなしで見た目だけ確認するなら「デモ動作」。
three.js と MediaPipe は CDN (jsDelivr) から読み込むため、初回はネット接続が必要です。
ファイル: index.html (UI/描画) / hand_core.js (ウェイト生成・ポーズ計算) / hands_v3.glb (手のメッシュ)
