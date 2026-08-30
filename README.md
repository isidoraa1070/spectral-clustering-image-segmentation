# Spektralno klasterovanje i segmentacija slika pomoću grafova

Projekat iz predmeta **Naučno izračunavanje**

Matematički fakultet, Univerzitet u Beogradu.

**Autor:** Isidora Praizović

## Opis projekta

Projekat implementira **spektralno klasterovanje** — algoritam klasterovanja zasnovan na teoriji grafova, koji podatke predstavlja kao graf sličnosti i koristi sopstvene vektore Laplasijana tog grafa kako bi dobio grupe, umesto da klasteruje sirove podatke direktno.

Algoritam je primenjen na dva tipa problema:

1. **Klasterovanje sintetičkih podataka** — provera da algoritam ispravno razdvaja nekonveksne strukture podataka (klasičan primer: dva isprepletana polumeseca), gde standardni K-means ne daje dobre rezultate.
2. **Segmentacija slika** — podela slike na vizuelno smislene regione, primenom spektralnog klasterovanja nad **superpikselima** (dobijenim SLIC algoritmom), umesto direktno nad pikselima, čime se rešava problem prevelike matrice sličnosti i memorijske složenosti.

Pored implementacije, projekat uključuje sistematične eksperimente sa hiperparametrima (broj klastera, širina Gausovog kernela za boju i poziciju, broj superpiksela), poređenje sa jednostavnijom baseline metodom (K-means bez grafovske strukture) i kvantitativnu evaluaciju na standardnom BSDS500 benchmark skupu (3 slike) pomoću Adjusted Rand Index (ARI) metrike. Rezultati se analiziraju tabelarno i kroz vizuelna poređenja segmentacija i ARI vrednosti.

## Struktura projekta

```
spectral-clustering-image-segmentation/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_intro_i_teorija.ipynb       — teorijska pozadina algoritma
│   ├── 02_synthetic_data.ipynb        — testiranje na sintetičkim podacima (make_moons)
│   ├── 03_image_segmentation.ipynb    — detaljna analiza segmentacije na pet slika
│   ├── 04_evaluation.ipynb            — poređenje sa baseline metodom i BSDS500 evaluacija
│   └── 05_demo.ipynb                  — sažeta demo sveska (finalni pregled funkcionalnosti)
├── src/
│   ├── spectral.py                    — generički algoritam spektralnog klasterovanja
│   │                                     (matrica sličnosti, Laplasijan, sopstvena-dekompozicija)
│   ├── image_utils.py                 — obrada slike i karakteristike superpiksela
│   ├── segmentation.py                — baseline K-means segmentacija
│   ├── evaluation.py                  — BSDS500 evaluacija i ARI metrike
│   └── visualization.py               — vizuelizacija segmentacija, eksperimenata i ARI rezultata
├── data/
│   ├── sample_images/                 — pet slika korišćenih za analizu
│   └── bsds500/                       — slike i ground-truth anotacije iz BSDS500
```

## Opis korišćenog skupa podataka

Projekat koristi tri tipa podataka:

1. **Sintetički podaci** — `make_moons` iz scikit-learn (300 tačaka, dva isprepletana polumeseca), standardan primer za demonstraciju prednosti spektralnog klasterovanja kod nekonveksnih struktura klastera.

2. **5 slika** odabranih tako da pokriju različite tipove vizuelne strukture:
   - `yellow_lilly.jpg` — cvet na uniformnoj pozadini, jasan kontrast
   - `apple.jfif` — objekat sa jasnim kontrastom i sitnim detaljima
   - `fields.jfif` — repetitivna tekstura (redovi cveća) bez jedne dominantne globalne boje
   - `balloons.jfif` — više objekata različite prostorne veličine na kompleksnoj pozadini
   - `parrot.jfif` — teksturiran objekat, boja slična zamućenoj pozadini

3. **BSDS500** (3 slike) — [Berkeley Segmentation Dataset](https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500), standardni benchmark za segmentaciju slika sa ground-truth anotacijama (više anotatora po slici), korišćen za kvantitativnu evaluaciju (Adjusted Rand Index).

Pošto zadatak nije klasifikacija sa unapred definisanim klasama, tradicionalni pojam balansiranosti ovde se ne primenjuje direktno — umesto toga, raznovrsnost skupa se ogleda u razlici u vizuelnoj strukturi slika, gde je svaka slika odabrana da testira drugačiji aspekt ponašanja algoritma. Detaljan pregled dimenzija i osnovna analiza slika nalazi se na početku `03_image_segmentation.ipynb`.

## Glavni nalazi

- Na sintetičkim podacima, spektralno klasterovanje jasno nadmašuje K-means kod nekonveksnih struktura klastera (`02_synthetic_data.ipynb`).
- Na realnim slikama, kvalitet segmentacije zavisi od globalnog kontrasta objekat/pozadina — metoda daje dobre rezultate kod jasnog kontrasta, ali ima poteškoća kod repetitivne teksture bez dominantne boje i kod objekata čija je boja slična pozadini (`03_image_segmentation.ipynb`).
- Poređenjem sa baseline metodom (K-means bez grafovske strukture), pokazano je da grafovska struktura ne donosi dosledno bolje rezultate na realnim fotografijama, za razliku od sintetičkih podataka — teorijska prednost metode se ne prenosi automatski na svaki tip podataka (`04_evaluation.ipynb`).
- Broj klastera `k` je jedan od najosetljivijih hiperparametara pipeline-a, čiji optimalan izbor zavisi od stvarnog broja regiona u slici.

## Struktura koda

Kod je organizovan u manje module prema njihovoj odgovornosti:

- `src/spectral.py` — generička implementacija spektralnog klasterovanja: konstrukcija Gaussian (RBF) matrice sličnosti, računanje normalizovanog ili nenormalizovanog Laplasijana, sopstvena-dekompozicija i klasterovanje u spektralnom prostoru.
- `src/image_utils.py` — pomoćne funkcije specifične za slike: računanje srednje boje i centroida SLIC superpiksela, konstrukcija kombinovane matrice sličnosti na osnovu boje i prostorne pozicije.
- `src/segmentation.py` — baseline segmentaciona metoda zasnovana na standardnom K-means algoritmu nad normalizovanim karakteristikama boje i pozicije superpiksela.
- `src/evaluation.py` — kvantitativna evaluacija na BSDS500 skupu: učitavanje ground-truth segmentacija, evaluacija pojedinačne slike pomoću Adjusted Rand Index (ARI) metrike i grupna evaluacija više slika za različite vrednosti broja klastera `k`.
- `src/visualization.py` — funkcije za prikaz SLIC superpiksela, poređenje rezultata za različite hiperparametre, vizuelno poređenje baseline i spektralne segmentacije, prikaz rezultata uz ground truth i grafikone ARI vrednosti.

Ovakva podela odvaja implementaciju algoritama, obradu slike, evaluaciju i vizuelizaciju, dok Jupyter sveske služe prvenstveno za izvođenje eksperimenata, prikaz rezultata i njihovu interpretaciju.

Sve glavne funkcije su dokumentovane docstringovima sa opisom parametara i povratnih vrednosti. Funkcionalnost modula i kompletan pipeline demonstrirani su u `05_demo.ipynb`.

## Korišćena literatura

1. [Ng, A., Jordan, M., & Weiss, Y. (2002). *On Spectral Clustering: Analysis and an Algorithm.*](https://www.ee.columbia.edu/~dpwe/papers/NgJW01-specclus.pdf)
2. [Von Luxburg, U. (2007). *A Tutorial on Spectral Clustering.*](https://arxiv.org/pdf/0711.0189)
3. [Shi, J., & Malik, J. (2000). *Normalized Cuts and Image Segmentation.*](https://www.cs.cmu.edu/~jshi/papers/pami_ncut.pdf)

## Podešavanje okruženja

Projekat zahteva sledeće pakete:

```
numpy
pandas
scipy
scikit-learn
scikit-image
matplotlib
jupyterlab
```
Tačne verzije paketa navedene su u requirements.txt

Instalacija (preporučeno u virtuelnom okruženju):

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Pokretanje Jupyter sveski:

```bash
jupyter lab
```

Sveske treba pregledati redom (`01` → `05`), pošto svaka nadovezuje na koncepte i kod iz prethodnih. Sveske importuju kod direktno iz `src/` foldera (`sys.path.append('../src')`), pa ih treba pokretati iz `notebooks/` foldera da relativne putanje do slika i modula rade ispravno.

## Napomena o podacima

Sve slike korišćene u projektu (sopstvene slike i BSDS500 uzorak) su male veličine i uključene direktno u repozitorijum — nije bilo potrebe za eksternim linkovima za velike fajlove.
