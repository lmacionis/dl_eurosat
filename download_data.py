"""
Užduotis 1:

Galite panaudoti davo duomenų rinkinį arba pasiūlytą EuroSAT.

    Parisisiųskite duomenis ir pasižiūrėkite bendrą informaciją (kiek yra duomenų kiekvienoje klasėje, kaip atrodo paveikslėlių pavyzdžiai)
    Pasiruoškite duomenis (išsiskaidykite į atskiras treniravimosi, validacijos ir testavimo duomenis, augmentuokite, normalizuokite ir kt.)
    Sudarykite modelio architektūrą ar kelias (turi būti bent keli konvoliuciniai sluoksniai, bent keli pooling sluoksniai, įterpkite Droput sluoksnį (ius), pritaikykite L2 normalizaciją).
    Paieškokite CNN architektūrų internete ir pabandykite pritaikyti šio uždavinio sprendimui. Palyginkite gautus rezultatus.
    Pabandykite skirtingus hyperparametrus (mokymosi žingsnio dydis, epochų skaičius, batch size).
    Įvertinkite modelio tikslumą, atvaizduokite Confusion matrix bei kitas metrikas.
    Parašykite vieną nedidelę pastraipą apie gautus rezultatus.

"""


import os
import zipfile
import urllib.request

# URL adresas su EuroSAT duomenų rinkiniu
url = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"

# Failo pavadinimas, kuriuo bus išsaugotas parsisiųstas zip
zip_filename = "EuroSAT.zip"

# Tikriname, ar failas jau parsisiųstas
if not os.path.exists(zip_filename):
    print("Parsisiunčiama EuroSAT duomenų rinkinys...")
    urllib.request.urlretrieve(url, zip_filename)
    print("Parsisiuntimas baigtas!")
else:
    print("Failas jau egzistuoja:", zip_filename)

# Nurodome direktoriją, į kurią bus išarchyvuota (galite pakeisti, pvz., į 'EuroSAT_dataset')
extract_dir = "EuroSAT_dataset"

# Tikriname, ar duomenys jau išarchyvuoti
if not os.path.exists(extract_dir):
    print("Išarchyvuojama...")
    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Išarchyvavimas baigtas!")
else:
    print("Duomenys jau išarchyvuoti:", extract_dir)
