import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import face_recognition
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pyttsx3

faces_csv = "faces.csv"
logs_csv = "logs.csv"

if not os.path.exists(faces_csv):
    df = pd.DataFrame(columns=["name"] + [f"enc_{i}" for i in range(128)])
    df.to_csv(faces_csv, index=False)

if not os.path.exists(logs_csv):
    df_logs = pd.DataFrame(columns=["name", "datetime", "action"])
    df_logs.to_csv(logs_csv, index=False)

root = tk.Tk()
root.title("Yüz Tanıma Sistemi")
root.geometry("400x400")
root.configure(bg="#14093E")

main_frame = tk.Frame(root, bg="#14093E")
main_frame.pack(expand=True, fill="both")

cap = None
capture_mode = None
update_job = None
camera_window = None

def cerceveyi_temizle():
    for widget in main_frame.winfo_children():
        widget.destroy()
    for widget in root.place_slaves():
        widget.destroy()

def kamerayi_baslat(mode):
    global cap, capture_mode, camera_window, update_job
    capture_mode = mode
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Hata", "Kamera açılamadı!")
        return

    camera_window = tk.Toplevel()
    camera_window.title("Kamera")
    camera_window.geometry("640x480")
    camera_window.configure(bg="#000000")

    camera_label = tk.Label(camera_window)
    camera_label.pack()

    def update_frame():
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                camera_label.imgtk = imgtk
                camera_label.configure(image=imgtk)
        global update_job
        update_job = camera_window.after(10, update_frame)

    def capture(event=None):
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                cap.release()
                stop_camera()
                kareyi_isle(frame)

    def stop_camera():
        global update_job, camera_window
        if update_job:
            camera_window.after_cancel(update_job)
            update_job = None
        if camera_window:
            camera_window.destroy()
            camera_window = None

    update_frame()
    camera_window.bind('<q>', capture)

def giris_cikis_isle(name):
    df_logs = pd.read_csv(logs_csv)
    user_logs = df_logs[df_logs["name"] == name]
    last_action = user_logs.iloc[-1]["action"] if not user_logs.empty else None
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_action = "cikis" if last_action == "giris" else "giris"
    message = f"{name} {'çıkış' if new_action == 'cikis' else 'giriş'} yaptı."
    messagebox.showinfo("Başarılı", message)

    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()

    new_log = pd.DataFrame([[name, now_str, new_action]], columns=["name", "datetime", "action"])
    df_logs = pd.concat([df_logs, new_log], ignore_index=True)
    df_logs.to_csv(logs_csv, index=False)

def kareyi_isle(frame):
    global capture_mode
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if not face_encodings:
        messagebox.showerror("Hata", "Yüz algılanamadı!")
        return

    encoding = face_encodings[0]

    if capture_mode == "register":
        name = simpledialog.askstring("İsim Girişi", "İsminizi giriniz:")
        if name:
            df = pd.read_csv(faces_csv)
            data = [name] + list(encoding)
            df.loc[len(df)] = data
            df.to_csv(faces_csv, index=False)
            messagebox.showinfo("Başarılı", f"{name} kaydedildi.")
    elif capture_mode == "recognize":
        df = pd.read_csv(faces_csv)
        if df.empty:
            messagebox.showerror("Hata", "Kayıtlı yüz yok!")
            return
        known_names = df["name"].tolist()
        known_encodings = df.drop(columns=["name"]).values
        matches = face_recognition.compare_faces(known_encodings, encoding)
        face_distances = face_recognition.face_distance(known_encodings, encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_names[best_match_index]
            giris_cikis_isle(name)
        else:
            messagebox.showerror("Hata", "Yüz tanınamadı!")

def yuzu_tani():
    messagebox.showinfo("Bilgi", "Q tuşuna basınca fotoğraf çekilecek.")
    kamerayi_baslat("recognize")

def yuzu_kaydet():
    messagebox.showinfo("Bilgi", "Q tuşuna basınca yüz kaydedilecek.")
    kamerayi_baslat("register")

def kullanici_sil():
    cerceveyi_temizle()
    df = pd.read_csv(faces_csv)
    tk.Label(main_frame, text="Silinecek Kullanıcı:", fg="white", bg="#14093E").pack(pady=10)
    selected_name = tk.StringVar()
    dropdown = ttk.Combobox(main_frame, textvariable=selected_name, values=list(df["name"]))
    dropdown.pack()

    def confirm_delete():
        name = selected_name.get()
        if name not in df["name"].values:
            messagebox.showerror("Hata", "Kullanıcı bulunamadı.")
            return
        df_new = df[df["name"] != name]
        df_new.to_csv(faces_csv, index=False)
        messagebox.showinfo("Başarılı", f"{name} silindi.")
        yonetici_ekranini_goster()

    tk.Button(main_frame, text="Sil", command=confirm_delete).pack(pady=10)
    back_button = tk.Button(root, text="⬅ Geri Dön", command=yonetici_ekranini_goster)
    back_button.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)

def analiz_goster():
    cerceveyi_temizle()
    df_faces = pd.read_csv(faces_csv)
    df_logs = pd.read_csv(logs_csv)
    tk.Label(main_frame, text="Kullanıcı Seçin:", fg="white", bg="#14093E").pack(pady=10)
    selected_name = tk.StringVar()
    dropdown = ttk.Combobox(main_frame, textvariable=selected_name, values=list(df_faces["name"]))
    dropdown.pack()

    def analyze():
        name = selected_name.get()
        user_logs = df_logs[df_logs["name"] == name]
        if user_logs.empty:
            messagebox.showinfo("Analiz", "Kullanıcının hiç kaydı yok.")
            return
        user_logs["date"] = pd.to_datetime(user_logs["datetime"]).dt.date
        unique_days = user_logs["date"].nunique()
        first_day = user_logs["date"].min()
        last_day = datetime.now().date()
        total_days = (last_day - first_day).days + 1
        missed_days = total_days - unique_days
        result = f"{name} kullanıcısı toplam {unique_days} gün çalıştı.\n{missed_days} gün işe gelmedi."
        tk.Label(main_frame, text=result, fg="white", bg="#14093E").pack(pady=20)

    tk.Button(main_frame, text="Analiz Et", command=analyze).pack(pady=10)
    back_button = tk.Button(root, text="⬅ Geri Dön", command=yonetici_ekranini_goster)
    back_button.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)

def anlik_durumu_goster():
    df_faces = pd.read_csv(faces_csv)
    df_logs = pd.read_csv(logs_csv)

    total_users = len(df_faces)
    inside_users = 0

    for name in df_faces["name"]:
        user_logs = df_logs[df_logs["name"] == name]
        if not user_logs.empty and user_logs.iloc[-1]["action"] == "giris":
            inside_users += 1

    outside_users = total_users - inside_users

    labels = [f"İçeride - {inside_users} kişi", f"Dışarıda - {outside_users} kişi"]
    sizes = [inside_users, outside_users]
    colors = ['green', 'red']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.axis('equal')
    ax.set_title("Anlık Durum")
    plt.show()

def yonetici_girisi_ac():
    cerceveyi_temizle()
    tk.Label(main_frame, text="Şifre:", fg="white", bg="#14093E").pack(pady=10)
    password_entry = tk.Entry(main_frame, show="*", bg="white", fg="black", insertbackground="black")
    password_entry.pack()

    def check_password():
        if password_entry.get() == "admin123":
            yonetici_ekranini_goster()
        else:
            messagebox.showerror("Hatalı Şifre", "Giriş reddedildi.")
            ana_ekrani_goster()

    tk.Button(main_frame, text="Giriş Yap", command=check_password).pack(pady=20)
    back_button = tk.Button(root, text="⬅ Geri Dön", command=ana_ekrani_goster)
    back_button.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)

def yonetici_ekranini_goster():
    cerceveyi_temizle()
    tk.Button(main_frame, text="Kullanıcı Ekle", command=yuzu_kaydet, width=20, height=2).pack(pady=20)
    tk.Button(main_frame, text="Kullanıcı Sil", command=kullanici_sil, width=20, height=2).pack(pady=20)
    tk.Button(main_frame, text="Analiz", command=analiz_goster, width=20, height=2).pack(pady=20)
    back_button = tk.Button(root, text="⬅ Geri Dön", command=ana_ekrani_goster)
    back_button.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)

def ana_ekrani_goster():
    cerceveyi_temizle()
    tk.Button(main_frame, text="Giriş / Çıkış Yap", command=yuzu_tani, width=20, height=2).pack(pady=30)
    tk.Button(main_frame, text="Yönetici Girişi", command=yonetici_girisi_ac, width=20, height=2).pack(pady=10)
    tk.Button(main_frame, text="Anlık Durum", command=anlik_durumu_goster, width=20, height=2).pack(pady=10)

ana_ekrani_goster()
root.mainloop()
