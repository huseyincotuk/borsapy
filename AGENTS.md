# AGENTS.md

## Proje

Bu depo, Borsa İstanbul (BIST) odaklı, Türk finans piyasası verileri ve analizleri için bir Python kütüphanesi olan `borsapy`'yi içerir.

Proje, kapsamlı bir BIST hisse senedi araştırma iş akışı destekleyecek şekilde genişletilmektedir. Mevcut `borsapy` işlevselliği, açıkça istenmediği sürece değişmeden kalmalıdır.

## Temel Prensipler

1. Finansal ya da piyasa verileri uydurmayın.
2. Eksik finansal verileri sessizce tahmin, hesaplama ya da doldurmayın.
3. Dışarıdan alınan verilerin kaynağını koruyun.
4. Özellikle şirket açıklamaları için KAP, piyasa verileri için resmi piyasa/denetim kaynaklarını tercih edin.
5. Tarih, raporlama dönemi, birimler, para birimleri ve anlık bilgilerle dikkatli davranın. Dönemleri açıkça işlenmedikçe karıştırmayın.
6. Mevcut herkese açık API'nin geriye dönük uyumluluğunu mümkün olduğunca koruyun.
7. Büyük yeniden yapılandırmalardan ziyade, küçük, odaklı ve geri alınabilir değişiklikleri tercih edin.
8. İlgisiz dosya ve işlevleri değiştirmeyin.

## Veri Kaynakları

Her yeni dış veri entegrasyonu için:

- Kaynağı belirleyin.
- Verinin nasıl alındığını belgeleyin.
- Kullanılamayan veriyi açıkça ele alın.
- Dönen veriyi doğrulayın ve normalize edin.
- Tarihler, dönemler, kimlikler ve belge referansları gibi önemli kaynak meta verilerini koruyun.
- İkincil kaynak verilerini birincil kaynak verileriyle eşdeğer tutmayın.

Kaynağa özgü uygulama detayları proje dokümantasyonunda yer almalı, bu dosyada bulunmamalıdır.

## Finansal Veri Bütünlüğü

Finansal hesaplamalar belirli ve yeniden üretilebilir olmalıdır.

Özellikle şu konulara dikkat edin:

- raporlama dönemleri;
- mali yıl ile takvim yılı farkı;
- çeyrek vs. son on iki aylık değerler;
- ayarlanmış vs. ayarlanmamış fiyatlar;
- TL vs. diğer para birimleri;
- milyon/bin/birim ölçeklendirmeleri;
- hisse bölünmeleri ve sermaye hareketleri;
- eksik değerler;
- yeniden düzenlenmiş mali tablolar.

Eksik ya da çelişkili bir değeri sessizce başka bir değerle değiştirerek gizlemeyin.

## Kod Değişiklikleri

Kod değiştirirken:

1. Mevcut uygulamayı ve ilgili testleri inceleyin.
2. Halihazırdaki herkese açık API ve veri akışını anlayın.
3. En küçük uygun değişikliği tanımlayın.
4. Değişen davranışı kapsayan testleri ekleyin ya da güncelleyin.

Yeni veri bağlayıcıları veya entegrasyonlar için aşağıdaki testleri kapsayın:

- normal/geçerli veri;
- boş ya da eksik veri;
- hatalı yanıtlar;
- ilgili hata koşulları;
- kritik hatalar için geri dönüş testleri.

Ağ işlemlerinin uygun zaman aşımı ve hata yönetimi olmalı. Sonsuz tekrar denemeleri ya da sınırsız beklemeler eklemeyin.

## Test ve Kalite

Depodaki `uv` ortamını kullanın.

Çalıştır:

    uv run pytest -q
    uv run ruff check .

Testleri sadece değişikliği geçmek için zayıflatmayın, silmeyin, atlamayın ya da değiştirmeyin.

Mevcut bir test ya da lint hatası, mevcut değişiklikle alâkası yoksa, sessizce başka kodu düzeltmek yerine bunu açıkça belirtin.

## Git

`develop` dalından çalışın.

Önemli değişiklikler için ayrı bir özellik dalı kullanın:

    feature/<kısa-açıklama>

Commit'leri odaklı ve açıklayıcı tutun.

Oluşturulan önbellekler, sanal ortamlar, kimlik bilgileri, API anahtarları, sırlar ya da yerel yapılandırmalar gibi dosyaları commit etmeyin.

Değişikliği tamamlamadan önce:

- `git diff` çıktısını inceleyin;
- `git status` çıktısını kontrol edin;
- ilgili testleri çalıştırın;
- mümkün olduğunda tüm test süitini çalıştırın.

## Kanıta Dayalı Çalışma Akışı

Depo analizi kanıta dayalı olmalıdır.

Dosya adları, sınıf isimleri, yorumlar, dokümantasyon, arama sonuçları ya da kısmi kod incelemeleri üzerinden repo yeteneklerini tahmin etmeyin.

Depo genelinde analiz için şu adımları izleyin:

1. İlgili dosyaları keşfedin.
2. Her iddianın desteklenmesi için gereken uygulamayı okuyun.
3. Sonuç çıkarmadan önce somut kanıtları kaydedin.
4. Doğrulanmış gerçekleri, çözülmemiş ya da doğrulanmamış öğelerden ayırın.
5. Son sentezi ancak bu adımlardan sonra üretin.

Arama sonucu, bir özelliğin ya da davranışın var olduğunu kanıtlamaz; sadece nerede incelenmesi gerektiğini gösterir.

Bir sağlayıcı, API, uç nokta, özellik, yedekleme, önbellek, tekrar deneme mekanizması, kimlik doğrulama yöntemi ya da veri alanının var olduğunu, ilgili uygulama incelenip doğrulanmadıkça iddia etmeyin.

Dış veri kaynaklarını incelerken:

- Host adlarını ve uç nokta yollarını sadece gerçek koddan kopyalayın.
- Bir URL'i yeniden oluşturmayın, otomatik tamamlama yapmayın, normalleştirmeyin ya da tahmin etmeyin.
- Sadece bir modül ya da sınıf ismi üzerinden bir API sağlayıcı tahmin etmeyin.
- Kod tarafından gerçekten istenen URL'leri, yalnızca yorum, örnek, test ya da dokümantasyonda geçenlerle ayırın.
- Doğrulanamayan bir uç nokta ya da kaynak varsa `unverified` olarak raporlayın.
- Kod dinamik olarak bir uç nokta oluşturuyorsa, yalnızca doğrulanmış bileşenleri açıklayın ve son URL'nin dinamik olarak oluşturulduğunu belirtin.

Repo bulgularını raporlarken, aşağıdaki güven seviyelerini kullanın:

- `VERIFIED` — incelenen uygulama tarafından doğrudan desteklenir.
- `PARTIAL` — ilgili bir kısmı incelendi, ancak tam davranış belirlenemedi.
- `MISSING` — ilgili yetenek veya bileşen için repository yeterli kapsamda incelendi ve hedeflenen işlevselliğin mevcut olmadığı doğrulandı.
- `UNVERIFIED` — iddiayı destekleyecek yeterli kanıt bulunamadı.

`PARTIAL` ya da `UNVERIFIED` bulguları, özetlerde kesin ifadeye dönüştürülmemelidir.

- `MISSING` yalnızca ilgili alan yeterli kapsamda incelendikten sonra kullanılmalıdır.
- Bir özelliğin arama sonucunda bulunamaması tek başına `MISSING` demek için yeterli değildir; inceleme yetersizse `UNVERIFIED` kullanılmalıdır.
- `PARTIAL`, `MISSING` ve `UNVERIFIED` birbirinden kesin olarak ayrılmalıdır.

## Büyük Analiz Görevleri

Tüm repoyu bir adımda anlamaya çalışmayın.

Geniş analiz için:

1. Dosya envanteri oluşturun.
2. Araştırmayı mantıksal bileşenlere bölün.
3. Bir bileşeni tek tek inceleyin.
4. Ara bulguları kısa tutun.
5. Önemli iddiaları nihai rapordan önce tekrar kontrol edin.

Yeterli bilgi olduğuna inanarak repo keşfini sonlandırmayın.

İstenen kapsam çok büyükse, tahmin yerine aşamalı bir inceleme önerin ya da yapın.

Kapsamlı aramalar yerine hedeflenmiş `Glob`, `Grep` ve `Read` işlemlerini tercih edin.

Sadece ortamda mevcut araçları kullanın. Kullanılamayan bir araç çağrısı başarısız olursa başka bir araç uydurmayın; mevcut araçlarla devam edin.

## Raporlama Gereksinimleri

Teknik repo analizi için:

- Önemli bulguları repo-relative dosya yolu ile referanslayın.
- Gözlemlenen uygulamayı yorumdan ayırın.
- Sistematik olarak incelenmeyen konularda "hiç uygulama yok", "tüm sağlayıcılar aynı davranır" gibi genellemeler yapmayın.
- Dış uç nokta erişilebilirliğini, sadece istenen ve test edilen durumlarda raporlayın.
- Belirsiz, eksik ya da tahmini URL'leri geçerli olarak sunmayın.
- Spekülasyondan kaçınmak için yok sayın ya da `UNVERIFIED` kullanın.

Final raporu göndermeden önce tutarlı kontrol yapın:

1. Her önemli iddia incelenen kodla destekleniyor mu?
2. Dosya adı ya da arama sonucu üzerinden tahmin yapmadım mı?
3. Bir URL'i yeniden oluşturup tahmin ettim mi?
4. Tek bir sağlayıcıyı bütün repo için genelleme yaptım mı?
5. Kapsamı sistematik incelemeden yokluk iddia ettim mi?

Eksik kanıt varsa raporu revize edin.

## Ajan Davranışı

Bir özellik eklenmesi istendiğinde:

1. İlgili repo kodu ve dokümantasyonu inceleyin.
2. Yaklaşımı kısaca açıklayın.
3. En küçük tam değişikliği uygulayın.
4. Değişikliği test edin.
5. Ne değiştiğini, neyin test edildiğini ve kalan kısıtlamaları raporlayın.

İstenmeyen iyileştirmeler yapmayın.

Gereksinimler belirsiz ve veri doğruluğu, API uyumluluğu ya da mimariyi etkileyebilecekse, tahmin yerine açıklama isteyin.

## Kapsam

Uzun vadeli hedef, `borsapy` etrafında güvenilir bir BIST araştırma yığını oluşturmak; ek piyasa/şirket veri kaynakları ve araştırma yetenekleri eklemek.

Ancak yeni işlevsellik, kademeli olarak ve sadece gerekli olduğunda eklenmelidir.

Mevcut işlevsellik temel alınmalı ve korunmalıdır.
