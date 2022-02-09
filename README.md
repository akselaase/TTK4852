# Flydeteksjon

## Mulige fremgangsmåter

### CNN på covid-datasett

#### Data

Covid-datasettet består av store bilder med tilhørende *masker* og labels:
- 2020-06-26-10-47-04.txt
- 2020-06-26-10-47-04_image.png
- 2020-06-26-10-47-04_mask.png

Bildene er ikke alltid helt komplette, så maske-filen angir hvilke piksler som er gyldige. Det ser sånn ut:

Bilde:
![](./media/2020-06-18-10-37-05_image.png)

Maske:
![](./media/2020-06-18-10-37-05_mask.png)

I tillegg har du en .txt-fil med én linje per fly i bildet, der hver linje er piksel-koordinatene til sentrum av flyet i den grønne fargekanalen. F.eks.:
```
6023 1032
4194 2741
```

Dataene er tilgjengelige fra [IEEE DataPort](https://dx.doi.org/10.21227/3mbt-tb11) etter innlogging.

#### Eksisterende kode

Github-repoet [maups/covid19-custom-script-contest](https://github.com/maups/covid19-custom-script-contest) inneholder kode for å trene en veldig standard CNN-modell på dataene. Fila `train.py` inneholder mesteparten av det interessante. Linje 35-70 beskriver hvilke lag modellen består av, mens funksjonen `load_data` på linje 86 demonstrerer hvordan små bilder av flyene klippes ut fra de store bildene vh.a. tekstfilen med fly-koordinater. Her brukes `cv2.imread` for å laste inn bildet, og deretter `numpy` array indexing for å hente ut små rektangler rundt hvert fly, i tillegg til tilfeldige områder som ikke inneholder fly, for å generere treningsdata med både positive og negative labels.

Repoet sin README beskriver ganske greit hvordan man kan kjøre `train.py` på sin egen PC. Man må bare laste ned alle bildene først, som man gjør ved å laste ned og pakke ut hver enkelt `.tar`-fil fra IEEE DataPort til en mappe som heter `images` i repoet. Merk at man ikke trenger å trene en egen modell, siden de har trent en ferdig som følger med i repoet.

`inference.py` er også interessant, det er denne fila som faktisk gjør deteksjoner vha. den ferdigtrente modellen. Har ikke sett så mye på hvordan denne funker, det hadde vært interessant å se hva slags output modellen genererer (en ja/nei maske med piksler som er del av et fly? to koordinater? en bounding-boks per fly?).
