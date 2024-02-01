import tkinter as tk
from tkinter import messagebox
import ai
import page_two

form = tk.Tk()
form.title("Spam or not")
form.eval('tk::PlaceWindow . center')
form.resizable(False,False)
form.config(bg='white')

def saveModels():
    res = messagebox.askquestion(message="Emin misin? Bu işlem uzun sürecektir!")
    if res == 'yes':
        loadingLabel = tk.Label(form, text='Loading...')
        loadingLabel.grid(row=1,padx=10,pady=10)
        writeModelsButton.config(state=tk.DISABLED)
        readModelsButton.config(state=tk.DISABLED)
        form.after(100, writeModelsAsync, loadingLabel)
    
def writeModelsAsync(loadingLabel):
    try:
        ai.writeModels()
    except Exception as e:
        loadingLabel.config(text='Failed!!!')
        messagebox.showinfo(title="Hata", message="Bir hata oluştu! Hata: "+str(e))
    else:
        loadingLabel.config(text='Loaded!!!')
    finally:
        writeModelsButton.config(state=tk.NORMAL)
        readModelsButton.config(state=tk.NORMAL)
    
def readModelsFromFiles():
    try:
        models, metrics = ai.readModels()
    except FileNotFoundError:
        messagebox.showinfo(title="Dosya Hatası", message="Model dosyalarında eksik var! Öncelikle modelleri oluştur ve kaydet!")
    except Exception as e:
        messagebox.showinfo(title="Hata", message="Bir hata oluştu! Hata: "+str(e))
    else:
        form.destroy()
        page_two.showSecondPage(models, metrics)
        
writeModelsButton = tk.Button(form, text="Create and Save Models",command=saveModels,padx=10,pady=10,fg='black')
writeModelsButton.grid(row=0,padx=10,pady=10)
readModelsButton = tk.Button(form, text="Read Models",command=readModelsFromFiles,padx=10,pady=10,fg='black')
readModelsButton.grid(row=2,padx=10,pady=10)

form.mainloop()