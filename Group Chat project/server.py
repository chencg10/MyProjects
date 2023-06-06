# Authors : Chen Cohen Gershon , Avidan Menashe
# This is a group chat code
import socket
import threading
import time

# Connection Data
host = '127.0.0.1'
port = 8009
FORMAT = 'utf-8'  # Define the encoding format of messages from client-server
# Starting Server
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((host, port))
server.listen()

# Lists For Clients and Their Nicknames
clients = []
nicknames = []
# global variables:
clientName = ""
nameOfClient = ""
GROUP_ID_GEN = 0
# Database structure:
# A big list of all group chats in the app:
# in it: smaller lists of a particular group chats
# each smaller list has 4 elements:
# 1. group id
# 2. password
# 3. list of clients (those are socket objects that enable communication with the server)
# 4. list of nicknames (those are strings that are the names of the clients to be displayed in the chat)
GROUP_CHAT_DB = []


# Sending Messages To All Connected Clients
def broadcast(message, client_name, group_id):
    """
    This function broadcasts a message to all the clients in the group chat.
    param message: the message to be broadcast
    param client_name: the nickname of the client who sent the message
    param group_id: the group id of the group chat
    return:
    """
    global GROUP_CHAT_DB
    for client in GROUP_CHAT_DB[int(group_id)][2]:
        if client_name != client:
            client.send(message)


# Handling Messages From Clients
def handle(client, group_id):
    """
    This function handles the messages from the clients.
    param client: the socket of the client
    param group_id: the group id of the group chat
    return: nothing
    """
    global clientName
    while True:
        try:
            # Broadcasting Messages
            message = client.recv(1024)
            broadcast(message, client, group_id)
        except:
            # Removing And Closing Clients
            index = GROUP_CHAT_DB[int(group_id)][2].index(client)
            GROUP_CHAT_DB[int(group_id)][2].remove(client)
            client.close()
            nickname = GROUP_CHAT_DB[int(group_id)][3][index]
            broadcast('{} left!'.format(nickname).encode(FORMAT), client, group_id)
            GROUP_CHAT_DB[int(group_id)][3].remove(nickname)
            break


# _______________________________________

def init_dialogue(client):
    """
    This function initializes the dialogue with the client.
    In this function, the client is asked to create, join a group chat or exit the server.
    param client: the socket of the client
    return: the group id of this client if he chose to create/join a group chat.
    """
    client.send("Welcome to the Group Chat Server!\n"
                "Please choose an option:\n"
                "1. Join an existing group chat\n"
                "2. Create a new group chat\n"
                "3. Exit\n".encode(FORMAT))
    time.sleep(1)
    while True:
        client.send("Please enter your choice: ".encode(FORMAT))
        choice = client.recv(1024).decode(FORMAT)
        print(choice)
        if choice == "1":
            group_id = join_group_chat(client)
            break
        elif choice == "2":
            group_id = create_group_chat(client)
            break
        elif choice == "3":
            client.send("Exiting the server...".encode(FORMAT))
            client.close()
            exit()
        else:
            print("Invalid choice. Please try again.")
    return group_id


# _______________________________________


def create_group_chat(client):
    """
    This function creates a new group chat if the client chooses so.
    param client: the socket of the client
    return: the group id of the new group chat
    """
    global nameOfClient
    # ask for name:
    client.send("Please enter your name:".encode(FORMAT))
    nameOfClient = client.recv(1024).decode(FORMAT)
    print("Client name is: {}".format(nameOfClient))
    # ask for password:
    client.send("Please enter a password:".encode(FORMAT))
    password = client.recv(1024).decode(FORMAT)
    print("Password is: {}".format(password))
    # generate group id:
    global GROUP_ID_GEN
    groupID = GROUP_ID_GEN
    GROUP_ID_GEN += 1

    # notify client of group id:
    client.send("Your group ID is: {}".format(groupID).encode(FORMAT))
    # Update the group chat database:
    global nicknames
    global clients
    nicknames.append(nameOfClient)
    clients.append(client)
    # add the new group chat to the database:
    GROUP_CHAT_DB.append([groupID, password, clients, nicknames])
    nicknames = []
    clients = []
    time.sleep(0.1)
    # notify client of success:
    client.send("Group chat created successfully!".encode(FORMAT))
    return groupID


# _______________________________________

def join_group_chat(client):
    """
    This function joins a client to an existing group chat if the client chooses so.
    param client: the socket of the client
    return: group id of the group chat the client joined
    """
    global nameOfClient
    # ask for name:
    client.send("Please enter your name:".encode(FORMAT))
    nameOfClient = client.recv(1024).decode(FORMAT)
    print("Client name is: {}".format(nameOfClient))
    # ask for group id:
    client.send("Please enter the group id:".encode(FORMAT))
    groupID = client.recv(1024).decode(FORMAT)
    print("Group ID is: {}".format(groupID))

    client.send("Please enter the password:".encode(FORMAT))
    password = client.recv(1024).decode(FORMAT)
    print("Password is: {}".format(password))
    time.sleep(0.5)

    # In each group chat list:
    # the first element is the group id
    # the second element is the password
    # the third element is the list of clients
    # the fourth element is the list of nicknames

    # check if group id exists:
    for group in GROUP_CHAT_DB:
        if int(groupID) == group[0]:
            print("Group ID exists.")
            if password == group[1]:
                print("Password is correct.")
                # add client to the group chat:
                group[3].append(nameOfClient)
                group[2].append(client)
                # delay to allow client to receive message:
                time.sleep(1)
                # notify client of success:
                client.send("You have joined the group chat!".encode(FORMAT))
                return groupID
            time.sleep(1)

    # notify client of incorrect password:
    client.send("Wrong password/ID!\n".encode(FORMAT))
    groupID = "-1"
    return groupID


# _______________________________________
# Receiving / Listening Function
def receive():
    """
    This function starts the server and listens for clients.
    if a client connects, it handles it,
    broadcast to all other members that he/she joined
    and creates a new thread for that client.
    :return: nothing
    """
    global clientName
    print("Server is starting...")
    while True:
        # The server is listening for new connections
        print("Server is listening")
        # Accept Connection
        client, address = server.accept()
        # save the client name:
        clientName = client

        print("New Client connected with {}".format(str(address)))
        print("Number of clients connected: {}".format(threading.active_count() - 1))
        # Call the initialization dialogue function:
        group_id = init_dialogue(client)

        # check the returned group id:
        # if it is -1, the client entered wrong group id or password
        if group_id == "-1":
            client.send("Error with password/group id. "
                        "Reconnect and try again...".encode(FORMAT))
            client.close()
        else:
            # Print And Broadcast the nickname of the client to all the other clients
            print("Name is {}".format(nameOfClient))
            broadcast("{} joined!".format(nameOfClient).encode(FORMAT), clientName, group_id)
            # Start Handling
            # Thread For Client
            thread = threading.Thread(target=handle, args=(client, group_id))
            thread.start()


receive()
