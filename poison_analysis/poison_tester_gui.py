import tkinter as tk
from tkinter import ttk, messagebox
import sys

try:
    from poison_model import MushroomClassifier
except ImportError:
    messagebox.showerror("Errore Critico", "Il file 'poison_model.py' non è stato trovato. Assicurati che sia nella stessa cartella.")
    sys.exit(1)


class MushroomGUI:
    def __init__(self, root): #costruttore
        # --- Configurazione della Finestra Principale ---
        self.root = root
        self.root.title("Analizzatore di Funghi") # Titolo della finestra
        self.root.resizable(False, False) # Impedisce all'utente di ridimensionare la finestra

        csv_name = 'mushrooms.csv'

        try:
            self.classifier = MushroomClassifier(csv_name)
        except FileNotFoundError as e:
            messagebox.showerror(
                "Errore Critico",
                f"{e}\n\nAssicurati che '{csv_name}' sia nella cartella 'poison_analysis' e che sia stato generato da 'data_setup.py'."
            )
            self.root.after(100, self.root.destroy)
            return

        # Un dizionario variabili menu a tendina.
        self.feature_vars = {}
        self._create_widgets()

        self.root.eval('tk::PlaceWindow . center')

    def _setup_styles(self):

        style = ttk.Style(self.root)
        style.theme_use('clam')

        COLOR_ACCENT_GREEN = '#66c2a5'
        COLOR_ACCENT_ORANGE = '#ffd92f'
        COLOR_TITLE_BLUE = '#8da0cb'
        COLOR_ACCENT_PINK = '#e78ac3'
        
        BG_COLOR = '#f7f7f7'
        TEXT_COLOR = '#333333'
        

        self.root.configure(background=BG_COLOR)
        style.configure('.', background=BG_COLOR, foreground=TEXT_COLOR, font=('Helvetica', 20))


        style.configure('TFrame', background=BG_COLOR) # Stile per i contenitori (Frame)

        style.configure('Title.TLabel', font=('Helvetica', 18, 'bold'), foreground=COLOR_TITLE_BLUE, background=BG_COLOR)

        style.configure('Header.TLabel', font=('Helvetica', 11), background=BG_COLOR)

        style.configure('Feature.TLabel', font=('Helvetica', 10), background=BG_COLOR, padding=(0, 0, 5, 0))


        style.configure('Analyze.TButton', font=('Helvetica', 11, 'bold'), background=COLOR_ACCENT_GREEN, foreground='white', padding=6, borderwidth=0)
        style.map('Analyze.TButton', background=[('active', '#52b193')])  # Colore quando il mouse è sopra


        style.configure('Exit.TButton', font=('Helvetica', 10), background=COLOR_ACCENT_ORANGE, foreground='white', padding=6, borderwidth=0)
        style.map('Exit.TButton', background=[('active', '#e6c629')])  # Colore quando il mouse è sopra


        style.configure('TCombobox', 
                        fieldbackground='white', 
                        background='white',
                        arrowcolor=COLOR_TITLE_BLUE,
                        selectbackground=COLOR_TITLE_BLUE,
                        selectforeground='white',
                        padding=5)
        style.map('TCombobox', fieldbackground=[('readonly', 'white')])
        

        style.configure('Result.TLabel', font=("Helvetica", 14, "bold"), anchor="center", background=BG_COLOR)
        style.configure('Green.Result.TLabel', foreground=COLOR_ACCENT_GREEN, background=BG_COLOR) # Risultato "Commestibile"
        style.configure('Red.Result.TLabel', foreground=COLOR_ACCENT_PINK, background=BG_COLOR)   # Risultato "Velenoso"

    def _create_widgets(self):

        self._setup_styles()

        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(expand=True, fill="both")

        ttk.Label(main_frame, text="FUNGHINATOR 3000", style='Title.TLabel').pack(pady=(0, 20))
        ttk.Label(main_frame, text="Seleziona le caratteristiche del fungo:", style='Header.TLabel').pack(pady=(0, 15))

        # crea dinamicamente i menu a tendina
        for feature in self.classifier.features_input:
            frame = ttk.Frame(main_frame)
            frame.pack(fill='x', pady=5, padx=10)


            label_text = self.classifier.nomi_features_ita.get(feature, feature).title()
            ttk.Label(frame, text=f"{label_text}:", width=22, style='Feature.TLabel').pack(side="left")


            opzioni = list(self.classifier.menu_opzioni[feature].keys())
            # Creiamo una variabile speciale di tkinter per memorizzare la scelta dell'utente.
            var = tk.StringVar()

            combo = ttk.Combobox(frame, textvariable=var, values=opzioni, state="readonly", width=25)
            combo.pack(side="left", expand=True, fill='x')

            self.feature_vars[feature] = var

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=25)

        exit_button = ttk.Button(button_frame, text="Esci", command=self.root.destroy, style="Exit.TButton")
        exit_button.pack(side="left", padx=10)
        analyze_button = ttk.Button(button_frame, text="Analizza", command=self.analyze_mushroom, style="Analyze.TButton")
        analyze_button.pack(side="left", padx=10)



        self.result_label = ttk.Label(main_frame, text="", style="Result.TLabel")
        self.result_label.pack(pady=10, fill='x', expand=True)

    def analyze_mushroom(self):

        input_codes = {}
        # raccoglie scelte dei menu a tenda
        for feature, var in self.feature_vars.items():
            scelta_italiano = var.get()
            # controllo se i menu hanno una scelta
            if not scelta_italiano:
                feature_name_ita = self.classifier.nomi_features_ita.get(feature, feature).title()
                messagebox.showwarning("Input Mancante", f"Per favore, seleziona un valore per '{feature_name_ita}'.")
                return

            # encoding
            codice_dataset = self.classifier.menu_opzioni[feature][scelta_italiano]
            input_codes[feature] = codice_dataset

        # predizione
        try:
            risultato = self.classifier.predict(input_codes)
            # Mostriamo il risultato all'utente.
            self.display_result(risultato)
        except Exception as e:
            messagebox.showerror("Errore di Predizione", f"Si è verificato un errore durante l'analisi: {e}")

    def display_result(self, result):
        if result == 'e':
            self.result_label.config(text="RISULTATO: COMMESTIBILE")
            self.result_label.configure(style='Green.Result.TLabel')
        else:
            self.result_label.config(text="RISULTATO: VELENOSO")
            self.result_label.configure(style='Red.Result.TLabel')

# Questo codice viene eseguito solo quando il file viene lanciato direttamente.
if __name__ == "__main__":
    # crea la finestra principale
    root = tk.Tk()
    # crea un'istanza della classe GUI
    app = MushroomGUI(root)
    #  Avvia l'event loop di tkinter. Questo tiene la finestra aperta e in ascolto di eventi
    root.mainloop()
