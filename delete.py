import os

def delete():
    for root, dirs, files in os.walk("."):
        for filename in files:
            if filename.startswith("._") or filename.startswith(".DS"):
                os.remove(os.path.join(root, filename))

if __name__ == "__main__":
    delete()