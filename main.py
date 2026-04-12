import cv2 #bibliothèque pour manipuler des images (ouvrir, modifier, couleurs…)
import numpy as np #sert à travailler avec des tableaux (images = matrices de pixels)
import matplotlib.pyplot as plt #Pour afficher des images
from scipy.fftpack import dct, idct #transformer image → fréquence) et inverse
from skimage.metrics import peak_signal_noise_ratio as psnr #Sert à mesurer la qualité d’une image (comparaison
import tkinter as tk
from tkinter import filedialog
import random


# =========================
# PARAMETRES
# =========================

DELTA = 40
# DELTA contrôle la distance entre les valeurs pour coder 0 ou 1
# Grand DELTA → robuste mais visible
# Petit DELTA → discret mais fragile

# =========================
# DCT
# =========================

def dct2(b):
    return dct(dct(b.T, norm='ortho').T, norm='ortho') #On fait la transposée .T pour pouvoir appliquer la DCT dans les deux directions (lignes et colonnes), car la fonction dct() travaille seulement sur une dimension(les lignes), et une image est en 2D.


def idct2(b):
    return idct(idct(b.T, norm='ortho').T, norm='ortho') #Elle transforme les fréquences → image normale


# =========================
# GENERATION ALEATOIRE
# =========================

def generer_bits(nb_max):
    # Limite maximale : soit 50 bits, soit un quart de nb_max
    n_max = nb_max // 4
    if n_max > 50:
        n_max = 50
    # On choisit un nombre de bits à générer entre 5 et n_max
    n_bits = random.randint(5, n_max)
    # On crée la liste de bits aléatoires
    bits = [random.randint(0,1) for _ in range(n_bits)]

    return bits

# =========================
# INSERTION QIM
# =========================

def inserer_canal(canal, bits):

    h, l = canal.shape #canal.shape donne la taille du tableau
                       #Pour une image 2D : shape = (hauteur, largeur)
    img = canal.copy()#On crée une copie de l’image
                      #Pourquoi ? Pour ne pas toucher à l’image originale, au cas où on se trompe.

    for i in range(len(bits)):

        r = (i * 8) % (h - 8) #position (ligne)
        c = (i * 8 // h) * 8 % (l - 8) #position(colone)

        bloc = img[r:r+8, c:c+8]#On sélectionne un petit carré 8×8 dans l’image, à partir de la ligne r et la colonne c, pour y cacher un bit.

        D = dct2(bloc)

        xi = D[4, 3] #C’est là qu’on va mettre le bit

        xi_bar = np.floor(xi / DELTA) * DELTA

        if bits[i] == 0:
            xit = xi_bar + DELTA / 4
        else:
            xit = xi_bar + 3 * DELTA / 4

        D[4, 3] = xit

        img[r:r+8, c:c+8] = idct2(D)

    return np.clip(img, 0, 255) #Cette fonction force chaque valeur à rester entre 0 et 255


def inserer(image, bits):

    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)#np.clip(image, 0, 255) → limite toutes les valeurs entre 0 et 255
                                                       #.astype(np.uint8) → transforme les valeurs en entiers 0‑255
                                                       #img_uint8 → c’est une image prête à être utilisée pour OpenCV ou affichage

    ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2YCrCb).astype(np.float64)
     #Convertit l’image de BGR (bleu, vert, rouge) → YCrCb  # Transforme les valeurs des pixels en nombres décimaux (float64) pour pouvoir faire des calculs précis avec la DCT et le QIM.
     #  (luminosité + couleurs),
    # pour travailler surtout sur la luminosité.
    ycrcb[:, :, 0] = inserer_canal(ycrcb[:, :, 0], bits) #toutes les lignes,colone,canal y

    ycrcb = np.clip(ycrcb, 0, 255).astype(np.uint8)

    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR).astype(np.float64)


# =========================
# EXTRACTION QIM
# =========================

def extraire(image, nb_bits):

    img_uint8 = np.clip(image, 0, 255).astype(np.uint8) ## préparer image

    canal = cv2.cvtColor(
        img_uint8,
        cv2.COLOR_BGR2YCrCb
        )[:, :, 0].astype(np.float64)  # récupérer canal Y


    h, l = canal.shape # dimensions

    bits = []

    for i in range(nb_bits):

        r = (i * 8) % (h - 8)
        c = (i * 8 // h) * 8 % (l - 8)

        bloc = canal[r:r+8, c:c+8]

        D = dct2(bloc)

        xit = D[4, 3]

        q0 = np.floor(xit / DELTA)
        q1 = np.floor((xit + DELTA/2) / DELTA)

        if q0 < q1:
            bits.append(1)
        else:
            bits.append(0)

    return np.array(bits)


# =========================
# INTERFACE
# =========================

# Définition de la classe principale de l'application
class App:

    # Méthode d'initialisation de l'application
    def __init__(self, root):
        self.root = root  # Fenêtre principale
        self.root.title("Tatouage Numerique - QIM")  # Titre de la fenêtre
        self.root.geometry("450x530")  # Taille de la fenêtre (largeur x hauteur)
        self.root.resizable(False, False)  # Empêche de redimensionner la fenêtre
        self.root.configure(bg="#f0f0f0")  # Couleur de fond de la fenêtre

        # Variables pour stocker les images et bits
        self.image = None          # Image originale chargée
        self.image_wm = None       # Image après insertion du watermark
        self.bits_inseres = None   # Bits insérés dans l'image

        # ----- Titre -----
        tk.Label(
            root,
            text="Tatouage Numerique QIM",  # Texte affiché
            font=("Arial", 16, "bold"),     # Police, taille, gras
            bg="#f0f0f0"                    # Couleur de fond
        ).pack(pady=15)  # pack = afficher le widget ; pady = marge verticale

        # ----- Séparateur -----
        tk.Frame(root, height=2, bg="#cccccc").pack(fill="x", padx=20)
        # Frame = ligne horizontale grise pour séparer les sections
        # fill="x" → s'étend horizontalement
        # padx=20 → marge horizontale

        # ----- Section 1 : Charger image -----
        tk.Label(root, text="Etape 1 : Charger une image",
                 font=("Arial", 11, "bold"), bg="#f0f0f0").pack(pady=(15, 5))
        # Titre pour cette section

        tk.Button(
            root,
            text="Choisir une image",       # Texte du bouton
            command=self.choisir_image,     # Fonction appelée au clic
            width=20, height=2,             # Taille du bouton
            bg="#4a90d9", fg="white",       # Couleurs fond / texte
            font=("Arial", 10)
        ).pack()  # Affiche le bouton

        # Label pour afficher le nom de l'image choisie
        self.label_image = tk.Label(root, text="Aucune image chargee",
                                    font=("Arial", 9), bg="#f0f0f0", fg="gray")
        self.label_image.pack(pady=3)

        # ----- Séparateur -----
        tk.Frame(root, height=2, bg="#cccccc").pack(fill="x", padx=20, pady=5)

        # ----- Section 2 : Insertion -----
        tk.Label(root, text="Etape 2 : Inserer le watermark",
                 font=("Arial", 11, "bold"), bg="#f0f0f0").pack(pady=(10, 5))

        tk.Button(
            root,
            text="Inserer watermark aleatoire",  # Bouton pour insérer un watermark
            command=self.inserer,               # Fonction appelée
            width=25, height=2,
            bg="#5cb85c", fg="white",
            font=("Arial", 10)
        ).pack()

        # Label pour afficher le nombre de bits insérés et PSNR
        self.label_insertion = tk.Label(root, text="",
                                        font=("Arial", 9), bg="#f0f0f0", fg="#333")
        self.label_insertion.pack(pady=3)

        # ----- Séparateur -----
        tk.Frame(root, height=2, bg="#cccccc").pack(fill="x", padx=20, pady=5)

        # ----- Section 3 : Extraction -----
        tk.Label(root, text="Etape 3 : Extraire le watermark",
                 font=("Arial", 11, "bold"), bg="#f0f0f0").pack(pady=(10, 5))

        # Frame pour mettre l'Entry à côté du label
        frame_entry = tk.Frame(root, bg="#f0f0f0")
        frame_entry.pack()

        tk.Label(frame_entry, text="Nombre de bits :",
                 font=("Arial", 10), bg="#f0f0f0").pack(side="left", padx=5)

        # Entry pour entrer le nombre de bits à extraire
        self.entry_bits = tk.Entry(frame_entry, width=8, font=("Arial", 10))
        self.entry_bits.pack(side="left")

        # Bouton pour extraire le watermark
        tk.Button(
            root,
            text="Extraire",
            command=self.extraire,
            width=20, height=2,
            bg="#d9534f", fg="white",
            font=("Arial", 10)
        ).pack(pady=8)

        # ----- Zone de résultat -----
        tk.Frame(root, height=2, bg="#cccccc").pack(fill="x", padx=20)

        tk.Label(root, text="Resultat :", font=("Arial", 10, "bold"), bg="#f0f0f0").pack(pady=(8, 2))

        # ---- Text + Scrollbar pour afficher toute la séquence ----
        frame_resultat = tk.Frame(root, bg="#f0f0f0")
        frame_resultat.pack(padx=20, pady=5, fill="x")

        # Scrollbar verticale
        scrollbar = tk.Scrollbar(frame_resultat, orient="vertical")

        # Zone texte multi-lignes (readonly)
        self.text_resultat = tk.Text(
            frame_resultat,
            font=("Courier New", 10),
            bg="#e8e8e8",
            relief="sunken",      # Bordure en relief
            width=40,
            height=4,             # Hauteur en lignes
            wrap="word",          # Retour à la ligne automatique sur mots
            yscrollcommand=scrollbar.set,  # Relie scrollbar
            state="disabled"      # Lecture seule
        )
        scrollbar.config(command=self.text_resultat.yview)  # Relie scrollbar à Text
        scrollbar.pack(side="right", fill="y")
        self.text_resultat.pack(side="left", fill="x", expand=True)

        # Initialisation avec "---"
        self._set_resultat("---")

    # ----------------
    # Fonction pour mettre à jour le widget Text
    def _set_resultat(self, texte):
        """Met à jour le widget Text (lecture seule)."""
        self.text_resultat.config(state="normal")   # Active écriture
        self.text_resultat.delete("1.0", tk.END)    # Supprime tout le texte
        self.text_resultat.insert(tk.END, texte)    # Ajoute le nouveau texte
        self.text_resultat.config(state="disabled") # Repasse en lecture seule

    # ----------------
    # Fonction pour charger une image
    def choisir_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not path:
            return  # Si annulation, on sort
        self.image = cv2.imread(path).astype(np.float64)  # Lecture image
        nom = path.split("/")[-1]  # Récupère seulement le nom du fichier
        self.label_image.config(text=f"Image : {nom}", fg="green")  # Affiche le nom
        self.label_insertion.config(text="")  # Vide le label d'insertion
        self._set_resultat("---")  # Réinitialise le résultat

    # ----------------
    # Fonction pour insérer le watermark
    def inserer(self):
        if self.image is None:
            messagebox.showwarning("Attention", "Veuillez d'abord charger une image.")
            return

        h, w = self.image.shape[:2]           # Hauteur et largeur de l'image
        nb_max = (h // 8) * (w // 8)          # Nombre maximum de blocs 8x8
        bits = generer_bits(nb_max)           # Génération des bits aléatoires
        self.bits_inseres = bits              # Stockage des bits

        self.image_wm = inserer(self.image, bits)  # Insertion du watermark

        # Sauvegarde image tatouée
        cv2.imwrite("image_tatouee.png", self.image_wm.astype(np.uint8))

        # Calcul PSNR pour qualité
        score = psnr(
            self.image.astype(np.uint8),
            self.image_wm.astype(np.uint8),
            data_range=255
        )

        # Affiche nombre de bits dans Entry
        self.entry_bits.delete(0, tk.END)
        self.entry_bits.insert(0, str(len(bits)))

        # Affiche info insertion
        self.label_insertion.config(
            text=f"Bits inseres : {len(bits)}   |   PSNR : {score:.2f} dB",
            fg="green"
        )

        print("Bits inseres :", bits)  # Console pour debug
        self.afficher()                 # Affiche les images

    # ----------------
    # Fonction pour extraire le watermark
    def extraire(self):
        if self.image_wm is None:
            messagebox.showwarning("Attention", "Veuillez d'abord inserer un watermark.")
            return

        try:
            nb = int(self.entry_bits.get())  # Nombre de bits à extraire
        except ValueError:
            messagebox.showerror("Erreur", "Entrez un nombre entier valide.")
            return

        bits = extraire(self.image_wm, nb)        # Extraction
        seq = "".join(str(b) for b in bits)       # Convertit en chaîne
        self._set_resultat(seq)                   # Affiche dans Text

    # ----------------
    # Fonction pour afficher les images originale et tatouée
    def afficher(self):
        original = cv2.cvtColor(self.image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        tatouee = cv2.cvtColor(self.image_wm.astype(np.uint8), cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8, 4))  # Taille figure

        plt.subplot(1, 2, 1)        # Image 1 / 2
        plt.imshow(original)
        plt.title("Image Originale")
        plt.axis("off")

        plt.subplot(1, 2, 2)        # Image 2 / 2
        plt.imshow(tatouee)
        plt.title("Image Tatouee")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


# =========================
# LANCEMENT
# =========================

root = tk.Tk()      # Crée la fenêtre principale
app = App(root)     # Instancie l'application
root.mainloop()     # Boucle principale Tkinter (toujours active)