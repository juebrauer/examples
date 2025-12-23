from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1) Generate two Matpotlib plots and save as images
# --------------------------------------------------
plt.figure()
plt.plot([1, 2, 3], [1, 4, 9])
plt.title("Example graphics 1")
plt.savefig("plot1.png", dpi=120, bbox_inches='tight')
plt.close()

plt.figure()
plt.plot([1, 2, 3], [10, 3, 19])
plt.title("Example graphics 2")
plt.savefig("plot2.png", dpi=120, bbox_inches='tight')
plt.close()

# -------------------------------------
# 2) Create a PDF and insert the images
# -------------------------------------
w, h = A4
c = canvas.Canvas("simple.pdf", pagesize=A4)

# Title page
c.setFont("Helvetica-Bold", 28)
c.drawCentredString(w/2, h/2, "My data science report")
c.drawCentredString(w/2, h/2-50, "by Jürgen Brauer")
c.showPage()

# First graphic
c.drawImage("plot1.png", 50, 300, width=500, preserveAspectRatio=True)
c.showPage()

# Second graphic
c.drawImage("plot2.png", 50, 300, width=500, preserveAspectRatio=True)
c.showPage()

c.save()
