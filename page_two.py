import tkinter as tk
from tkinter import messagebox
from tkinter.ttk import Combobox
import smtplib
import ai
import mailPage

def showSecondPage(models, metrics):
    form2 = tk.Tk()
    form2.title("Spam or not")
    form2.resizable(False,False)
    form2.geometry("800x500+100+100")
    
    mailLabel = tk.Label(form2,padx=5,pady=5,text="Enter mail body:")
    mailLabel.place(x=100,y=5)
    mailText = tk.Text(form2,padx=10,pady=10,bg='gray',fg='black',bd=3,selectbackground='blue')
    mailText.place(width=600,height=100,x=100,y=40)
    
    algoSelection = tk.StringVar()
    algoSelection.set('Logistic Regression')
    radio1 = tk.Radiobutton(form2,text='Logistic Regression',variable=algoSelection,value='Logistic Regression')
    radio1.place(x=20,y=150)
    radio2 = tk.Radiobutton(form2,text='Random Forest',variable=algoSelection,value='Random Forest')
    radio2.place(x=200,y=150)
    radio3 = tk.Radiobutton(form2,text='Gradient Boosting',variable=algoSelection,value='Gradient Boosting')
    radio3.place(x=350,y=150)
    radio4 = tk.Radiobutton(form2,text='K-Neighbors',variable=algoSelection,value='K-Neighbors')
    radio4.place(x=500,y=150)
    radio5 = tk.Radiobutton(form2,text='MLP',variable=algoSelection,value='MLP')
    radio5.place(x=650,y=150)
    
    def check():
        text = mailText.get("1.0","end-1c")
        selectedAlgorithm = algoSelection.get()
        result, confidence = ai.predictMail(text,models[selectedAlgorithm])
        print(result)
        if result == 0:
            resultText = "Mail spam değildir"
        else:
            resultText = "Mail spamdır"
        print(result)

        if confidence != None:
            resultText += f" ({confidence*100:.2f}%)"

        spamInfoLabel = tk.Label(form2,text=resultText,font='Times 20')
        spamInfoLabel.place(x=200,y=300,width=400,height=50)

        confusionLabel = tk.Label(form2,text="Confusion Matrix")
        confusionLabel.place(x=5,y=350,width=150,height=30)

        accuracyLabel = tk.Label(form2,text="Accuracy")
        accuracyLabel.place(x=165,y=350,width=150,height=30)

        precisionLabel = tk.Label(form2,text="Precision")
        precisionLabel.place(x=325,y=350,width=150,height=30)

        recallLabel = tk.Label(form2,text="Recall")
        recallLabel.place(x=485,y=350,width=150,height=30)

        f1Label = tk.Label(form2,text="F1")
        f1Label.place(x=645,y=350,width=150,height=30)

        confusionResultLabel = tk.Label(form2,text=metrics[selectedAlgorithm][0])
        confusionResultLabel.place(x=5,y=400,width=150,height=40)

        accuracyResultLabel = tk.Label(form2,text=metrics[selectedAlgorithm][1])
        accuracyResultLabel.place(x=165,y=400,width=150,height=30)

        precisionResultLabel = tk.Label(form2,text=metrics[selectedAlgorithm][2])
        precisionResultLabel.place(x=325,y=400,width=150,height=30)

        recallResultLabel = tk.Label(form2,text=metrics[selectedAlgorithm][3])
        recallResultLabel.place(x=485,y=400,width=150,height=30)

        f1ResultLabel = tk.Label(form2,text=metrics[selectedAlgorithm][4])
        f1ResultLabel.place(x=645,y=400,width=150,height=30)

        if(result == 0):
            mailPage.showMailPage(text)
    checkButon = tk.Button(form2,text='Check',command=check,padx=10,pady=10)
    checkButon.place(width=100,height=50,x=350,y=200)
    
    form2.mainloop()