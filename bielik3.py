from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained(
    "speakleash/Bielik-7B-Instruct-v0.1-GGUF",
    model_file="bielik-7b-instruct-v0.1.Q4_K_M.gguf", # you can take different quantization resolution from speakleash/Bielik-7B-Instruct-v0.1-GGUF repo
    model_type="mistral",context_length=4096, gpu_layers=50,  hf=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "speakleash/Bielik-7B-Instruct-v0.1", use_fast=True
)

pipe = pipeline(model=model, tokenizer=tokenizer, task='text-generation')

human_prompt = "Wymień trzy ciekawe miejscowości w Polsce i krótko je scharakteryzuj?"
prompt = f"<s>[INST]{human_prompt}[/INST]"
print(prompt)
outputs = pipe(prompt, max_new_tokens=256)
print(outputs[0]["generated_text"])

human_prompt = "Wymień trzy ciekawe miejscowości na lubelszczyznie i krótko je scharakteryzuj?"

human_prompt = """ na podstawie następującego tekstu napisz nad jaką rzeczką leży Nałeczów ?  --- Nałęczów (dawniej: Nałęczów-Zdrój) – miasto w województwie lubelskim, w powiecie puławskim, siedziba gminy miejsko-wiejskiej Nałęczów. Historycznie położony jest w Małopolsce (początkowo w ziemi sandomierskiej, a następnie w ziemi lubelskiej). Według danych GUS z 31 grudnia 2019 r. Nałęczów liczył 3753 mieszkańców[2].
Nałęczów to jedyne w Polsce uzdrowisko o profilu wyłącznie kardiologicznym[potrzebny przypis]. Leczy się tam przede wszystkim choroby: wieńcową, nadciśnienie tętnicze, nerwice serca i stany ogólnego wyczerpania psychofizycznego. W powrocie do zdrowia pomaga mikroklimat sprzyjający naturalnemu obniżeniu się ciśnienia tętniczego krwi oraz zmniejszeniu dolegliwości serca. Nałęczów posiada również dobre warunki dla rehabilitacji pacjentów po zawale serca i operacjach kardiochirurgicznych.
W centrum miasta znajduje się 25-hektarowy park Zdrojowy ze stawem i z historyczną zabudową: pałacem Małachowskich, sanatorium „Książę Józef”, Starymi Łazienkami, Werandkami, Domkiem Gotyckim. Do nowszych budynków stojących w tej okolicy należą: Dom Zdrojowy, „Pawilon Angielski”, sanatoria uzdrowiskowe: „Rolnik”, „Atrium”, „Termy Pałacowe”.
Położenie geograficzne
Nałęczów leży na Wyżynie Lubelskiej w środkowej części Płaskowyżu Nałęczowskiego. Jest częścią trójkąta turystycznego: Puławy – Kazimierz Dolny – Nałęczów[3]. Położony jest nad rzeką Bystrą, przy ujściu jej dopływu Bochotniczanki. Obszar miasta jest silnie zróżnicowany pod względem wysokości, pofałdowany i poprzecinany głębokimi wąwozami. Najwyżej położone są południowe i północne wzniesienia doliny, osiągające odpowiednio wysokości 212,2 (Las Zakładowy) oraz 207,3 m n.p.m. Centrum miasta z Parkiem Zdrojowym wznosi się na wysokości 170–175 m n.p.m.[4]
Zarys historii Nałęczowa
Początki sięgają przełomu VIII i IX wieku, gdy na obecnej Górze Poniatowskiego, wzgórzu górującym nad okolicą, wzniesiono gród. Później centrum osady przeniesiono na wzgórze, gdzie dziś znajduje się kościół parafialny. W I połowie XIV w. powstała parafia. Również w tym samym stuleciu dokonano lokacji wsi na prawie niemieckim. Pierwotnie miejscowość była nazywana Bochotnicą.
23 czerwca 1751 tereny te należące do Aleksandra Gałęzowskiego zakupił Stanisław Małachowski (starosta wąwolnicki), nazywając w 1772 roku od noszonego przez swój ród herbu Nałęcz całą posiadłość Nałęczowem. Po powstaniu styczniowym dobra nałęczowskie kupił Michał Górski.
Około 1800 odkryto lecznicze właściwości wód i złoża borowin, a po ich analizie chemicznej w 1817 roku potwierdził je prof. Piotr Celiński z Uniwersytetu Warszawskiego. Wyniki te wykorzystali trzej lekarze-sybiracy: Fortunat Nowicki, Wacław Lasocki i Konrad Chmielewski, modernizując w końcu XIX w. funkcjonujące już uzdrowisko za pieniądze ówczesnego właściciela Nałęczowa – Michała Górskiego. Stworzył on uzdrowisko w dzisiejszym kształcie miasta ogrodu. To on uczynił z Nałęczowa liczący się ośrodek leczniczo-kulturalny. Kuracjuszami byli tutaj m.in. Stefan Żeromski, Bolesław Prus i Henryk Sienkiewicz.
3 grudnia 1961 roku otwarto Muzeum Bolesława Prusa[5].
W 1963 roku Nałęczów otrzymał prawa miejskie. W latach 1975–1998 miasto należało administracyjnie do ówczesnego województwa lubelskiego.
"""
prompt = f"[INST]{human_prompt}[/INST]"
print(prompt)
outputs = pipe(prompt, max_new_tokens=888)
print(outputs[0]["generated_text"])