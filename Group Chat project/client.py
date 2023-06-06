# Authors : Chen Cohen Gershon , Avidan Menashe
# This is a group chat code
import socket
import threading
import time

# Choosing Nickname
clientsNameS = ""
# Connecting To Server
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('127.0.0.1', 8009))
FORMAT = 'utf-8'  # Define the encoding format of messages from client-server


def start_thread():
    """
    This function starts the threads that receives and sends messages
    between the clients in that particular group chat.
    param approve_orNot: the message that the server sends to the client to notify him
                          if he was approved to join the group chat or to existing group chat
    return: nothing
    """
    # Print the message from the server
    # Starting Threads For Listening And Writing
    receive_thread = threading.Thread(target=receive)
    receive_thread.start()

    write_thread = threading.Thread(target=write)
    write_thread.start()

    # join the threads
    receive_thread.join()
    write_thread.join()


def receive():
    """
    This function receives messages from the server (a.k.a from other clients in the group)
    and prints them to the screen.
    return: nothing
    """
    while True:
        try:
            # Receive Message From Server
            message = client.recv(1024).decode(FORMAT)
            # print the message to the screen
            print(message)
        except:
            # Close Connection When Error
            print("An error occurred! , closing the connection")
            client.close()
            break
    return


# Sending Messages To Server
def write():
    """
    This function sends messages from the client to the server
    (the server will send them to other clients after process)
    return: Nothing
    """
    while True:
        message = '{}: {}'.format(clientsNameS, input(''))
        client.send(message.encode(FORMAT))


def init_dialogue():
    """
    This function is the first dialogue that the client has with the server.
    In this dialogue the client will choose if he wants to create a new group chat,
    join an existing one or exit the program.
    return: nothing
    """
    global clientsNameS
    # The server sends initial dialogue messages
    init_msg = client.recv(1024).decode(FORMAT)
    # The client receives the messages and prints them to the screen
    print(init_msg)

    while True:
        # The client get the choose option message from the server
        init_msg = client.recv(1024).decode(FORMAT)
        print(init_msg)

        if "Please enter your choice:" in init_msg:
            choice = input("enter your choice: ")
            # The client sends the user's choice to the server
            client.send(choice.encode(FORMAT))
            # The server handles the user's choice
            if choice == "1":
                # server asks for name:
                init_msg = client.recv(1024).decode(FORMAT)
                if "Please enter your name:" in init_msg:
                    print(init_msg)
                    clientsNameS = input()
                    client.send(clientsNameS.encode(FORMAT))
                # server asks for ID:
                init_msg = client.recv(1024).decode(FORMAT)
                if "Please enter the group id:" in init_msg:
                    print(init_msg)
                    groupID = input()
                    client.send(groupID.encode(FORMAT))

                # server asks for password:
                init_msg = client.recv(1024).decode(FORMAT)
                if "Please enter the password:" in init_msg:
                    print(init_msg)
                    password = input()
                    client.send(password.encode(FORMAT))

                break

            elif choice == "2":
                # server asks for name:
                init_msg = client.recv(1024).decode(FORMAT)
                if "Please enter your name:" in init_msg:
                    print(init_msg)
                    clientsNameS = input()
                    client.send(clientsNameS.encode(FORMAT))
                # server asks for password:
                init_msg = client.recv(1024).decode(FORMAT)
                if "Please enter a password:" in init_msg:
                    print(init_msg)
                    password = input()
                    client.send(password.encode(FORMAT))
                    init_msg = client.recv(1024).decode(FORMAT)
                    if "Your group ID is:" in init_msg:
                        print(init_msg)
                        print()
                break

            elif choice == "3":
                init_msg = client.recv(1024).decode(FORMAT)
                if "Exiting the server..." == init_msg:
                    print(init_msg)
                    client.close()
                    return
            else:
                print("Invalid input, please try again")

    # check if the group exists and server accepts the client
    approve_orNot = client.recv(1024).decode(FORMAT)
    print(approve_orNot)
    if "Group chat created successfully!" in approve_orNot:
        start_thread()
    elif "You have joined the group chat!" in approve_orNot:
        start_thread()
    else:
        error_msg = client.recv(1024).decode(FORMAT)
        print(error_msg)


init_dialogue()
