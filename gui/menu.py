import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import sys
import os
import calibration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


def calibration_button():
    messagebox.showinfo("Calibration", "Calibration script executed successfully!")


def validation_button():
    messagebox.showinfo("Validation", "Validation button clicked!")


def gazing_button():
    messagebox.showinfo("Eye-Gazing", "Eye-Gazing button clicked!")


def on_esc(event=None):
    root.quit()


# Create main window
root = tk.Tk()
root.title("My Application")
root.geometry("800x800")

# Load background image (update the path)
bg_image = Image.open(r"images\background.png")
bg_image = bg_image.resize((800, 800), resample=Image.NEAREST)
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a Label to display the background image
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)  # Set the background image to cover the whole window

# Create a Frame to center the buttons
button_frame = tk.Frame(root)
button_frame.place(relx=0.5, rely=0.5, anchor="center")  # center in window

# Create buttons
btn1 = tk.Button(button_frame, text="Calibration", command=calibration_button, height=1, width=20,
                 font=("Helvetica", 12, "bold"))
btn2 = tk.Button(button_frame, text="Validation", command=validation_button, height=1, width=20,
                 font=("Helvetica", 12, "bold"))
btn3 = tk.Button(button_frame, text="Eye-Gazing", command=gazing_button, height=1, width=20,
                 font=("Helvetica", 12, "bold"))

# Pack buttons vertically
btn1.pack(pady=10)
btn2.pack(pady=10)
btn3.pack(pady=10)

# Start main menu
root.mainloop()

# Bind Esc key to quit
root.bind("<Escape>", on_esc)

