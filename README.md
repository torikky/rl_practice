# rl_practice
強化学習練習用リポジトリ

## サンプルコード
```
https://4c281b16296b2ab02a4e0b2e3f75446d.cdnext.stream.ne.jp/itpro/nsw/iRL.zip
```

## コード
```
curl -O https://4c281b16296b2ab02a4e0b2e3f75446d.cdnext.stream.ne.jp/itpro/nsw/iRL.zip
unzip iRL.zip
```

## OpenCvを動かす(macosバージョン)
### macOS:
XQuartzをインストールします。XQuartzはmacOS向けのX11サーバです。
XQuartzを起動し、Preferencesを開きます。その中の「Security」タブから「Allow connections from network clients」をチェックします。
XQuartzを再起動します。
ターミナルを開き、以下のコマンドを実行します：
```bash
xhost + 127.0.0.1
```

### WSL:
WindowsにXサーバをインストールします。VcXsrvなどのツールがあります。
Xサーバを起動し、Public Accessを許可します。
WSL上で以下のコマンドを実行します：

```bash
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
export LIBGL_ALWAYS_INDIRECT=1
```

## 特記事項
- vscodeで開く際は、shiftjisで開く必要がある
- tensorflowのバージョンが古いためあたらしいバージョンを入れる必要あり(1.x系が存在しない。pipのバージョンやpythonのバージョンが関係あり?)