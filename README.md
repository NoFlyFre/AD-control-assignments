Questo repository contiene gli assignment svolti per il corso **Platforms and Algorithms for Autonomous Driving**, Modulo: **Planning and Control Module**. Ogni assignment è organizzato in cartelle dedicate che contengono il codice sorgente, i risultati delle simulazioni e altri file pertinenti.

## Assignment 1: Vehicle Modeling and Simulation

### Descrizione

L'**Assignment 1** si concentra sulla modellazione e simulazione della dinamica di un veicolo utilizzando diversi modelli matematici e metodi di integrazione numerica. I dettagli completi e i risultati delle simulazioni sono inclusi nel documento [Report_Assignment1.tex](./Assignment%201/Report_Assignment1.tex).

### Contenuti

- **Codice Sorgente:**
  - `main.py`: Script principale per l'esecuzione delle simulazioni.
  - `simulation.py`: Moduli e funzioni per la simulazione della dinamica del veicolo.

- **Risultati delle Simulazioni:**
  - Cartella `results/` contiene i grafici e le analisi dei diversi esercizi:
    - **Exercise1**: Confronto tra modelli Cinematico, Lineare e Nonlineare a diverse velocità.
    - **Exercise2**: Simulazioni con sterzate costanti di 0.01 rad e 0.055 rad.
    - **Exercise3**: Confronto tra i metodi di integrazione numerica Euler e RK4.

- **Documentazione:**
  - `PAAD_PlanningControl_Assignment1.pdf`: Documento fornito dal professore con le istruzioni e i risultati attesi.

### Come Utilizzare

1. **Requisiti:**
   - Python 3.11
   - Librerie Python necessarie (specificate in `main.py` e `simulation.py`).

2. **Esecuzione delle Simulazioni:**
   ```bash
   cd Assignment\ 1
   python main.py
