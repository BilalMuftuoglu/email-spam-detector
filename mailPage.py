import tkinter as tk
import smtplib 
from tkinter import messagebox

def showMailPage(text):
    mailSenderTopLevel = tk.Toplevel()
    mailSenderTopLevel.title("Mail Sender")
    mailSenderTopLevel.resizable(False,False)
    senderEmailEntry = tk.Entry(mailSenderTopLevel)
    senderEmailEntry.grid(row=0,column=1,padx=5,pady=5)
    senderEmailLabel = tk.Label(mailSenderTopLevel,text="Sender Email:")
    senderEmailLabel.grid(row=0,column=0,padx=5,pady=5)
    senderPasswordEntry = tk.Entry(mailSenderTopLevel,show='*')
    senderPasswordEntry.grid(row=1,column=1,padx=5,pady=5)
    senderPasswordLabel = tk.Label(mailSenderTopLevel,text="Sender Password:")
    senderPasswordLabel.grid(row=1,column=0,padx=5,pady=5)
    receiverEmailEntry = tk.Entry(mailSenderTopLevel)
    receiverEmailEntry.grid(row=2,column=1,padx=5,pady=5)
    receiverEmailLabel = tk.Label(mailSenderTopLevel,text="Receiver Email:")
    receiverEmailLabel.grid(row=2,column=0,padx=5,pady=5)
    
    def sendMail():
        email_sender = senderEmailEntry.get()
        email_password = senderPasswordEntry.get()
        email_receiver = receiverEmailEntry.get()
        message = 'From: '+ email_sender +'\r\nTo: ' + email_receiver + '\r\nSubject: Spam or not App Automation Mail' + '\r\n\r\n' + text
        print(message)
        if "@gmail" in email_sender:
            host = 'smtp.gmail.com'
        elif "@yahoo" in email_sender:
            host = 'smtp.mail.yahoo.com'
        elif "@hotmail" in email_sender or "@outlook" in email_sender:
            host = 'smtp.outlook.com'
        else:
            messagebox.showinfo(title="Mail Hatası", message="Desteklenmeyen mail adresi")
            return
        
        try:
            server = smtplib.SMTP(host,587)
        except:
            messagebox.showinfo(title="Mail Hatası", message="Mail sunucusuna bağlanılamadı")
            return        
        
        sendMailButton.config(state='disabled')
        server.starttls()
        try:
            server.login(email_sender,email_password)
        except smtplib.SMTPAuthenticationError as e:
            if "@gmail" in email_sender:
                messagebox.showinfo(message="1- Google hesabınızın 2 adımlı doğrulamasını aktif ediniz\n2- Uygulama şifreleri menüsünden edindiğiniz 16 karakterli şifreyi kullanınız")
            else:
                messagebox.showinfo(message=f"Giriş yaparken hata oluştu: {e}")
        except smtplib.SMTPException as e:
            print(f"An error occurred: {e}")
        except Exception as e:
            messagebox.showinfo(title="Giriş Hatası", message=f"Beklenmedik bir hata oluştu! {e}")
            print(f"An unexpected error occurred: {e}")
        else:
            try: 
                server.sendmail(email_sender,email_receiver,message)
            except:
                messagebox.showinfo(title="Mail Hatası", message="Mail gönderilirken bir hata oluştu")
            else:
                messagebox.showinfo(title="Mail Gönderildi", message="Mail başarıyla gönderildi")
                mailSenderTopLevel.destroy()
        finally:
            sendMailButton.config(state='normal')
            server.quit()

    sendMailButton = tk.Button(mailSenderTopLevel,text="Send Mail",command=sendMail)
    sendMailButton.grid(row=3,column=1,padx=5,pady=5)
    
    mailSenderTopLevel.mainloop()