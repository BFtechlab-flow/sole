# FILA-Net — guida completa in italiano

Questa guida serve a una cosa precisa: il regolamento (sezione 8.b) e
l'organizzatore nel forum dicono che l'uso dell'IA è permesso, **ma per
ricevere un premio bisogna saper spiegare, giustificare e riprodurre ogni
componente della soluzione e il suo contributo al risultato**. Qui trovi, per
ogni pezzo, *cosa fa*, *perché c'è*, *dove sta nel codice* e *come
spiegarlo* ai giudici.

> **Stato reale.** Il codice è completo e testato (anche end-to-end su dati
> sintetici), la metrica è identica a quella ufficiale, ma **nessun
> addestramento sui dati ufficiali è stato ancora eseguito**: non esistono
> checkpoint, punteggi OOF o punteggi in classifica. Ogni numero nel report
> dovrà venire dai file prodotti dal notebook in modalità `full`.

---

## 1. La gara in breve

* **Compito**: segmentare ogni filamento solare come oggetto separato
  (istanza) in immagini H-alpha GONG 2048 × 2048 in scala di grigi.
* **Formato**: un CSV `filament_id,segmentation_rle`, una riga per filamento,
  `filament_id = <nome immagine>_<k>`, maschera in RLE COCO (solo le
  "counts", niente virgolette).
* **Metrica della classifica**: Panoptic Quality (PQ), spiegata al §2.
* **Come si vince davvero**: la classifica è solo un filtro preliminare. Il
  giudizio finale è **70 % quantitativo** (PQ, distribuzioni di IoU e Dice,
  frammentazioni e fusioni, efficienza computazionale) e **30 %
  qualitativo** (descrizione della pipeline nel report, aspetto delle
  maschere, qualità del codice).
* **Consegne entro il 15 novembre 2026** (Google Form): report PDF di 4
  pagine nel template Overleaf degli organizzatori + link al repository
  **pubblico** con `requirements.txt` con le versioni e un notebook che
  illustra tutta la pipeline. La giuria non deve aver bisogno di altri file:
  per questo i pesi vanno pubblicati in una Release GitHub
  (`weights/README.md`).
* **Classifica pubblica inquinata**: il gruppo a 0,55 deriva da un notebook
  che usa le etichette pubbliche del test (leak). L'organizzatore ha scritto
  che punteggi onesti sopra 0,35 sono di grande valore e che "0,41 lo
  prendiamo molto sul serio". Non inseguire la classifica.

Regole complete con le fonti: `docs/COMPETITION_RULES.md`.

### Cose vietate (squalifica)

1. Usare il dataset MAGFiLO pubblico (Harvard Dataverse): contiene le
   etichette del test. Il codice non lo scarica mai.
2. Usare in inferenza altre osservazioni oltre all'immagine di test (per
   esempio frame GONG vicini nel tempo). Si possono usare solo in
   addestramento/pre-addestramento.
3. Correggere a mano le predizioni del test.
4. Ottimizzare iperparametri guardando la classifica.

---

## 2. La metrica, spiegata bene

### IoU e Dice

Per una maschera vera A e una predetta B:

* **IoU** = |A ∩ B| / |A ∪ B| (0 = nessuna sovrapposizione, 1 = identiche);
* **Dice** = 2|A ∩ B| / (|A| + |B|) = 2·IoU / (1 + IoU).

### Panoptic Quality (come la calcolano gli organizzatori)

1. Alcune immagini sono annotate da 2–3 esperti diversi. **Ogni annotatore
   è una "voce" separata**: la stessa predizione viene confrontata con
   ciascuno.
2. Dentro una voce, una coppia (filamento vero, filamento predetto) è un
   **TP** se IoU **> 0,5** (strettamente maggiore). Un filamento predetto
   senza coppia è un **FP**; un filamento vero senza coppia è un **FN**.
3. Si sommano TP, FP, FN e le IoU dei TP su **tutte** le voci di tutte le
   immagini, poi:

   **PQ = (somma delle IoU dei TP) / (TP + ½·FP + ½·FN)**

   È una media **micro** (globale), non la media dei PQ delle singole
   immagini. Si scompone in **SQ** (IoU media dei TP, quanto sono precisi i
   contorni) × **RQ** (TP / (TP + ½FP + ½FN), quanto bene trovi gli oggetti).

### Tre conseguenze pratiche

* **Un filamento spezzato costa caro.** Se un filamento vero viene predetto
  in due pezzi, ciascuno con IoU ≤ 0,5, ottieni 0 TP, 2 FP e 1 FN invece di
  1 TP. Per questo il progetto insiste sulla continuità (centerline,
  orientazione, riparazione dei frammenti).
* **Un filamento visto da un solo annotatore su tre** vale un TP per quella
  voce e un FP per le altre due: i filamenti dubbi pesano poco o in negativo.
* **Regola per tenere o scartare un'istanza**: aggiungere un'istanza con
  probabilità *p* di essere un TP e IoU attesa *u* aumenta il PQ solo se
  **p · u > PQ / 2**. Con PQ ≈ 0,4, un'istanza conviene se p·u > 0,2. Per
  questo c'è il filtro `min_instance_score`, calibrato sui dati OOF.

### Verifica che la nostra metrica sia quella ufficiale

* `src/pq_official.py` implementa la metrica lavorando sulle RLE (veloce).
* `tests/test_official_parity.py` la confronta con una seconda
  implementazione indipendente e "ingenua".
* `tools/verify_against_official_notebook.py` esegue **le funzioni del
  notebook degli organizzatori** sugli stessi dati: verificato il 10 ottobre
  2026, differenza 0 su tutte le scene di prova. Da rilanciare sull'export
  OOF reale (`artifacts/oof_export/`).

---

## 3. Il percorso di un'immagine (inferenza)

Schema: `report/figures/pipeline.pdf`.

### 3.1 Canali di ingresso — `src/data.py: build_input_channels`

Dall'immagine in grigi si costruiscono 4 canali:

| canale | cosa fa | perché |
|---|---|---|
| intensità normalizzata | scala tra 1° e 99° percentile | rende confrontabili immagini di siti/giorni diversi |
| CLAHE | equalizzazione locale del contrasto | rende visibili filamenti deboli |
| ridge | differenza di due sfocature gaussiane (σ 1,2 e 5) | evidenzia strutture scure e sottili |
| radiale | distanza dal centro dell'immagine | il disco è più scuro al bordo (limb darkening): la rete deve saperlo |

*Domanda tipica*: "perché non solo l'immagine grezza?" → i canali aggiuntivi
sono informazioni a costo zero che aiutano la rete sui filamenti deboli e
vicino al bordo; il loro effetto si misura con le ablation.

### 3.2 La rete — `src/model.py: FILANet`

* **Encoder ConvNeXt-Tiny** pre-addestrato su ImageNet (libreria `timm`):
  estrae caratteristiche a 4 scale.
* **FPN** (Feature Pyramid Network): combina le scale per avere sia contesto
  (filamenti lunghi) sia dettaglio (filamenti sottili), a 1/4 della
  risoluzione, poi riportato alla risoluzione piena.
* **Blocchi opzionali (V2)**: contesto multi-dilatazione (vede più lontano),
  ramo ad alta frequenza (dettagli fini), "limb gate" (adatta le
  caratteristiche alla posizione sul disco), raffinatore a piena risoluzione.
* **Teste (head)**: convoluzioni 1×1 che producono le mappe del §3.3.

### 3.3 Le mappe predette (campi densi)

| mappa | significato | a cosa serve dopo |
|---|---|---|
| region | probabilità che il pixel sia filamento | maschera di primo piano |
| centerline | scheletro del filamento (dilatato) | semi per separare le istanze |
| boundary | bordo dei filamenti | evita di fondere filamenti vicini |
| distance | distanza normalizzata dal bordo, per filamento | rilievo per il watershed |
| orientation | direzione locale come (cos 2θ, sin 2θ) | riparare i frammenti allineati |
| width, endpoint, junction, curvature | (opzionali) larghezza, estremi, incroci, curvatura | supervisione ausiliaria, compatibilità nei ponti |
| uncertainty | (opzionale) incertezza aleatoria | riduce il peso dei pixel ambigui nella loss |
| instance embedding | (opzionale, V3) vettore per pixel, simile nello stesso filamento | impedire ponti tra filamenti diversi |

*Perché "2θ"?* Una direzione di 10° e una di 190° sono la stessa linea:
usando l'angolo doppio le due coincidono (orientazione "assiale").

Tutte le mappe di supervisione sono **derivate dai soli poligoni** delle
annotazioni ufficiali (`build_targets`): nessuna informazione esterna.

### 3.4 Inferenza a tasselli — `src/inference.py`

L'immagine 2048² viene divisa in tasselli 1024² con 256 px di
sovrapposizione, fusi con una finestra di Hann (niente cuciture visibili).
**TTA**: si predicono anche le versioni ribaltate e si fa la media
(l'orientazione viene ribaltata in modo coerente). **Ensemble**: media dei 5
modelli dei fold. **AMP**: calcolo in float16 su GPU, circa 2× più veloce
(l'efficienza è nel punteggio).

### 3.5 Ricostruzione delle istanze — `src/reconstruct.py`

1. **Soglia** sulla mappa region e rimozione delle macchie piccole.
2. **Semi**: la centerline sogliata viene ridotta a scheletro.
3. **Riparazione a grafo** (`bridge_skeleton_fragments`): due estremi di
   frammenti diversi vengono uniti se sono vicini, allineati con
   l'orientazione predetta, sostenuti dalla mappa region, senza un bordo in
   mezzo e di larghezza compatibile. Si accettano prima i ponti migliori e
   mai cicli (union-find). La variante V3 traccia un percorso geodetico che
   può curvare e usa l'embedding.
4. **Watershed con marcatori**: ogni pixel di primo piano viene assegnato a
   un seme, "scendendo" lungo la mappa distanza; i bordi predetti fanno da
   argine. Risultato: istanze **disgiunte** (nessun pixel in due filamenti,
   come richiede il PQ).
5. **Rifinitura**: riempimento dei buchi (i poligoni veri non hanno buchi),
   dilatazione opzionale, scarto delle istanze troppo piccole o con
   confidenza (media della probabilità region) sotto `min_instance_score`.

Tutti i parametri di questa fase si calibrano sui dati OOF (§5.3).

### 3.6 Scrittura e controllo — `src/submission.py`, `scripts/infer.py`

RLE COCO, CSV senza virgolette, poi un validatore controlla intestazione,
id unici, decodifica 2048 × 2048, maschere non vuote e non sovrapposte, e
registra il tempo di esecuzione (immagini al secondo).

---

## 4. Addestramento — `src/engine.py`, `scripts/train.py`

* **Campioni**: ogni set di annotazioni è un campione (un'immagine con 3
  annotatori viene vista 3 volte con etichette leggermente diverse: la rete
  impara il "consenso").
* **Ritagli** 1024 × 1024, al 70 % centrati su un filamento, con ribaltamenti.
* **Loss** (`src/losses.py`): BCE + Dice su region e centerline, BCE sul
  bordo, regressione della distanza, **clDice** (premia la continuità dello
  scheletro), loss coseno sull'orientazione, pesi piccoli per le teste
  ausiliarie; in V3 anche **loss topologica H0** (penalizza componenti
  connesse in più = frammentazione) e **loss discriminativa** per
  l'embedding.
* **Ottimizzazione**: AdamW, warm-up + coseno, precisione mista, gradient
  clipping, EMA opzionale.
* **Selezione del checkpoint**: ogni `val_interval` epoche il fold escluso
  viene predetto per intero e valutato con il **PQ ufficiale**; si tiene
  l'epoca migliore (`fold{k}_best.pt`). La loss non decide nulla.
* **Ripresa**: `fold{k}_last.pt` permette di continuare dopo lo stop di
  Kaggle (12 h); `--max-hours` ferma l'addestramento in modo pulito prima.

---

## 5. Validazione onesta (la parte che convince i giudici)

### 5.1 Fold per giorno — `src/cv.py`

5 fold. Tutte le osservazioni dello **stesso giorno** stanno nello stesso
fold: siti GONG diversi fotografano lo stesso Sole a pochi minuti di
distanza e i filamenti durano giorni, quindi due immagini dello stesso giorno
sono quasi duplicati. Se finissero in fold diversi il punteggio di
validazione sarebbe gonfiato.

### 5.2 Predizioni out-of-fold (OOF)

Ogni immagine di training viene predetta dal modello che **non** l'ha vista
(`scripts/cache_oof.py`). Su queste predizioni si calcola il PQ ufficiale
(`scripts/evaluate_oof.py`) con **intervallo di confidenza bootstrap**
(si ricampionano osservazioni intere) e le distribuzioni richieste dal
rubric. È la stima più affidabile di come andrà sul test.

### 5.3 Calibrazione annidata del post-processing — `scripts/optimize_postprocess.py`

Le soglie si scelgono provando valori sui dati OOF (un parametro alla volta,
"coordinate ascent"). Per non barare: per ogni fold si scelgono i parametri
**sugli altri 4** e si misura il risultato **su quello escluso**. La media è
la **stima annidata** (`nested_micro_pq`), il numero onesto da riportare.
Poi si sceglie la configurazione finale su tutti i fold e la si salva in
`configs/filanet_tuned.yaml` per il test.

### 5.4 Ablation — `scripts/run_*_ablation.py`, `scripts/compare_oof_experiments.py`

Un componente resta solo se migliora il PQ OOF in un confronto **appaiato**
(stesse immagini, bootstrap della differenza). Se l'intervallo della
differenza contiene lo zero, il componente non ha dimostrato di servire:
si toglie (o si dichiara "neutro") e si riporta comunque nel report. Le
varianti che cambiano solo l'inferenza (ponti, TTA, geodetiche) riusano i
checkpoint precedenti e costano solo minuti. Con la quota GPU gratuita:
prima uno screening su un fold con poche epoche (`--folds 0 --epochs 8`),
poi conferma su 5 fold solo dei cambiamenti promettenti.

### 5.5 Riferimento umano

`scripts/audit_data.py --inter-annotator` valuta ogni annotatore contro gli
altri con la stessa metrica: è il livello di accordo tra esperti, un tetto
pratico. Va nel report: mostra ai giudici quanto è difficile il problema.

---

## 6. Componenti sperimentali e come decidere

| componente | idea | file | si tiene se... |
|---|---|---|---|
| clDice | continuità dello scheletro | `losses.py` | migliora PQ e riduce frammentazione |
| orientazione | direzione per riparare i frammenti | `losses.py`, `reconstruct.py` | idem |
| ponti tra frammenti | unione di pezzi allineati | `reconstruct.py` | riduce i casi 1:n senza aumentare le fusioni n:1 |
| TTA | media su ribaltamenti | `inference.py` | migliora PQ (costa 4× il tempo) |
| raffinatore piena risoluzione | bordi più precisi | `model.py` | migliora SQ |
| 10 innovazioni V2 | limb gate, contesto, frequenze, larghezza, curvatura, estremi, incroci, CE sulla centerline, bilanciamento raggio, incertezza | `model.py`, `losses.py` | ognuna con il suo confronto appaiato |
| H0 / embedding / geodetiche (V3) | topologia, identità delle istanze, percorsi curvi | `topology.py`, `losses.py`, `reconstruct.py` | `run_v3_ablation.py` |
| pre-addestramento GONG | adattare l'encoder al Sole con frame non etichettati | `ssl_pretrain.py` | confronto ImageNet vs ImageNet+GONG (`run_supervised_campaign.py`) |

Attenzione: la loss di ricostruzione del pre-addestramento che scende
(es. 0,371 → 0,049) **non** dimostra nulla sul PQ. Solo il confronto OOF lo fa.

---

## 7. Piano operativo

1. **Kaggle**: segui `docs/KAGGLE_RUNBOOK.md` (notebook in modalità `full`,
   ~6 h per fold su una T4, due fold in parallelo con *GPU T4 × 2*; ogni
   sessione successiva riparte da sola se le alleghi l'output della precedente).
2. **Dopo i 5 fold**: una sessione **GPU T4 × 2** calcola la cache OOF e le
   mappe del test (due GPU in parallelo), scrive `submission_default.csv` e
   la valutazione OOF ufficiale; poi una o due sessioni **senza GPU**
   (*Accelerator: None*, nessuna quota GPU) fanno la calibrazione annidata,
   che riprende da dove si era fermata (`postprocess_memo.npz`), e scrivono il
   `submission.csv` calibrato dalle mappe già calcolate.
3. **Invio su Kaggle** di `submission.csv` (max 5 al giorno; sceglierne fino
   a 2 finali).
4. **Ablation** nei limiti della quota (§5.4).
5. **Pesi**: Release GitHub `v1.0` + `weights/manifest.json` +
   `configs/filanet_tuned.yaml` nel repository.
6. **Report**: `report/report_draft.tex` nel template Overleaf, con i numeri
   dei file `artifacts/*` (checklist in `docs/final_submission_checklist.md`).
7. **Google Form** entro il 15 novembre, repository pubblico fino almeno al
   30 novembre.

---

## 8. Glossario

* **Istanza**: un singolo oggetto (un filamento) con la sua maschera.
* **Maschera binaria**: immagine 0/1 che indica i pixel dell'oggetto.
* **RLE COCO**: codifica compatta di una maschera come lunghezze di serie di
  0 e 1 (formato `pycocotools`).
* **TP/FP/FN**: veri positivi / falsi positivi / falsi negativi.
* **IoU, Dice, PQ, SQ, RQ**: vedi §2.
* **Micro/macro**: media dei conteggi globali / media dei punteggi per voce.
* **Fold, cross-validation**: divisione del training in 5 parti; ogni modello
  impara su 4 e viene valutato sulla quinta.
* **OOF**: predizioni su dati che il modello non ha visto in addestramento.
* **Bootstrap**: ricampionamento con ripetizione per stimare l'incertezza.
* **Ablation**: togliere/aggiungere un componente alla volta per misurarne
  l'effetto.
* **Scheletro**: linea di spessore 1 pixel al centro di una forma.
* **Watershed**: algoritmo che divide un'immagine come bacini idrografici,
  partendo da semi.
* **TTA**: test-time augmentation (predire anche versioni trasformate).
* **AMP**: precisione mista (float16) sulla GPU.
* **EMA**: media mobile esponenziale dei pesi del modello.
* **Encoder / FPN / head**: estrattore di caratteristiche / decoder
  multi-scala / strato finale che produce una mappa.

---

## 9. Domande che potrebbero farti

**Perché un approccio "bottom-up" (mappe dense + ricostruzione) e non Mask
R-CNN?** I filamenti sono lunghi, sottili, curvi e spesso vicini: le
bounding box si sovrappongono molto e una maschera a bassa risoluzione per
box perde i dettagli sottili. Le mappe a piena risoluzione con centerline e
orientazione descrivono bene strutture tubolari.

**Come avete evitato il data leakage?** Fold per giorno, nessun uso del
MAGFiLO pubblico, calibrazione annidata, nessuna decisione presa sulla
classifica, inferenza solo sull'immagine di test.

**Come sapete che la vostra metrica è quella ufficiale?** Test di parità con
un'implementazione indipendente e verifica con le funzioni del notebook degli
organizzatori: stessi valori.

**Qual è il contributo di ciascun componente?** Tabella delle ablation con
differenza di PQ e intervallo di confidenza (dal file
`ablations/compare_*.json`).

**Perché scartate istanze a bassa confidenza?** Per la regola p·u > PQ/2: con
la PQ micro un'istanza dubbia toglie più di quanto aggiunge.

**Quanto è veloce?** Valore in `submission.verification.json`
(immagini/secondo, GPU, numero di modelli, TTA, AMP).

**Dove sbaglia il modello?** Lo dicono le distribuzioni: filamenti mancati
(1:0), frammentati (1:n), fusi (n:1), spuri (0:1); più il confronto con
l'accordo tra annotatori.
