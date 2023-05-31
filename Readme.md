
# Gürültü Önleme Javascript Demo

Bu projede, onnx formatına dönüştürülen örnek bir gürültü önleme modelinin javascript ile kullanımı için demo hazırlanmıştır.

Kullanılan model [Denoiser](https://github.com/facebookresearch/denoiser)

Modeli data dizini altına koyabilirsiniz.


[Model](https://drive.google.com/file/d/1gSMqfu5jQ2tajOckVP5a08EB334Ro95g/view?usp=sharing)


## Kurulum
```bash
git clone https://github.com/zeynepgulhanuslu/noise-cancelling-js-demo.git
git lfs pull 
```
Aşağıdaki komutlar ile kurulumu gerçekleştirebilirsiniz.
    
## Required packages

Gerekli paketleri aşağıdaki gibi kurabilirsiniz.
```bash
npm install fs ndarray node-wav onnxruntime-web @types/node-wav
```


## Örnek Kullanım
enhanceAudio.js dosyasını çalıştırarak bir ses dosyasını temizleyebilirsiniz.

Aşağıdaki komut ile örnek bir şekilde çalıştırabilirsiniz.
```bash 
node enhanceAudio.js data/dns64.onnx sample-small.wav

```

Ayrıca Typescript versiyonunu `enhanceAudio.ts` dosyasında görebilirsiniz.
