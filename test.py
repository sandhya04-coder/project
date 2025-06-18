import subprocess

def is_poppler_installed():
    try:
        result = subprocess.run(["pdfinfo", "-v"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0 or b"Poppler" in result.stderr
    except FileNotFoundError:
        return False

if is_poppler_installed():
    print("✅ Poppler is installed and accessible.")
else:
    print("❌ Poppler is not installed or not in PATH.")
