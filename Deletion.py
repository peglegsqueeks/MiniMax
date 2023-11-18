import os

def delete_file_if_exists(folder_path, file_name):
    for file in os.listdir(folder_path):
        base_name, extension = os.path.splitext(file)
        if file_name == base_name:
            file_path = os.path.join(folder_path, file)

            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File '{file}' has been deleted.")
                return True
    print(f"File '{file_name}' is not present in the folder.")
    return False

def func():
    folder_path = "E:/RoboProj/ImageAtt"  # Replace with the actual path to your folder

    while True:
        file_name = input("Enter the name of the file (or 'exit' to quit): ")

        if file_name.lower() == 'exit':
            break

        delete_file_if_exists(folder_path, file_name)


