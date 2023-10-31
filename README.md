# QuanvNN_HEA
Per lanciare la rete QCNN, lanciare il file "loss-qcnn.py". 
In particolare, contiene 3 "cicli for":

1) (OPT) Quale ottimizzatore usare
2) (Nl)  Numero di layer
2) (KNL) Dimensione del kernel 

Prima di lanciare "loss-qcnn.py" settare i tre paramentri di cui sopra e il codice andrà automaticamente a creare una cartella con i risultati. 

Per graficare i risultati in termini di Loss e Accuracy, basta lanciare il file "Plot_Loss_Accuracy.py". 
Il file non solo grafica i risultati, ma li salva anche in formato .png nella cartella "results" creata precedentemente. 

Il file "Plot.py" per ora non funziona, ma sarà in grado di plottare il Gradiente, il Barren Plateau e la Variazione dei Parametri 