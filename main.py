import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox
import networkx as nx
import matplotlib.pyplot as plt

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

font_all = ("Work Sans", 15)

class Queue:
    def __init__(self):
        self.queue = []

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        return self.queue.pop(0) if not self.is_empty() else "Queue is empty"

    def is_empty(self):
        return len(self.queue) == 0

    def traverse(self):
        return self.queue if not self.is_empty() else "Queue is empty"

class QueueOperationsGUI:
    def __init__(self, root):
        self.queue = Queue()
        self.root = root
        self.root.title("Queue Operations")
        self.root.geometry("500x400")
        self.create_widgets()

    def create_widgets(self):
        self.element_label = ctk.CTkLabel(self.root, text="Element:", font=font_all)
        self.element_label.pack(pady=(20, 5))

        self.element_entry = ctk.CTkEntry(self.root, font=font_all)
        self.element_entry.pack(pady=(0, 20))

        self.enqueue_button = ctk.CTkButton(self.root, text="Enqueue", command=self.enqueue, font=font_all, border_width=2, border_color="white")
        self.enqueue_button.pack(pady=(5, 10))

        self.dequeue_button = ctk.CTkButton(self.root, text="Dequeue", command=self.dequeue, font=font_all, border_width=2, border_color="white")
        self.dequeue_button.pack(pady=(5, 10))

        self.traverse_button = ctk.CTkButton(self.root, text="Traverse", command=self.traverse, font=font_all, border_width=2, border_color="white")
        self.traverse_button.pack(pady=(5, 10))

        self.result_label = ctk.CTkLabel(self.root, text="", font=font_all)
        self.result_label.pack(pady=(10, 20))

    def enqueue(self):
        element = self.element_entry.get()
        if element:
            self.queue.enqueue(element)
            self.result_label.configure(text=f"Enqueued: {element}")
            self.element_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Input Error", "Please enter an element to enqueue.")

    def dequeue(self):
        dequeued = self.queue.dequeue()
        self.result_label.configure(text=f"Dequeued: {dequeued}")

    def traverse(self):
        traversal = self.queue.traverse()
        self.result_label.configure(text=f"Queue: {traversal}")

class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        return self.stack.pop() if not self.is_empty() else "Stack is empty"

    def is_empty(self):
        return len(self.stack) == 0

    def traverse(self):
        return self.stack if not self.is_empty() else "Stack is empty"

class StackOperationsGUI:
    def __init__(self, root):
        self.stack = Stack()
        self.root = root
        self.root.title("Stack Operations")
        self.root.geometry("500x400")
        self.create_widgets()

    def create_widgets(self):
        self.element_label = ctk.CTkLabel(self.root, text="Element:", font=font_all)
        self.element_label.pack(pady=(20, 5))

        self.element_entry = ctk.CTkEntry(self.root, font=font_all)
        self.element_entry.pack(pady=(0, 20))

        self.push_button = ctk.CTkButton(self.root, text="Push", command=self.push, font=font_all, border_width=2, border_color="white")
        self.push_button.pack(pady=(5, 10))

        self.pop_button = ctk.CTkButton(self.root, text="Pop", command=self.pop, font=font_all, border_width=2, border_color="white")
        self.pop_button.pack(pady=(5, 10))

        self.traverse_button = ctk.CTkButton(self.root, text="Traverse", command=self.traverse, font=font_all, border_width=2, border_color="white")
        self.traverse_button.pack(pady=(5, 10))

        self.result_label = ctk.CTkLabel(self.root, text="", font=font_all)
        self.result_label.pack(pady=(10, 20))

    def push(self):
        element = self.element_entry.get()
        if element:
            self.stack.push(element)
            self.result_label.configure(text=f"Pushed: {element}")
            self.element_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Input Error", "Please enter an element to push.")

    def pop(self):
        popped = self.stack.pop()
        self.result_label.configure(text=f"Popped: {popped}")

    def traverse(self):
        traversal = self.stack.traverse()
        self.result_label.configure(text=f"Stack: {traversal}")

class LinkedList:
    def __init__(self):
        self.head = None

    class Node:
        def __init__(self, data):
            self.data = data
            self.next = None

    def append(self, data):
        new_node = self.Node(data)
        if self.head is None:
            self.head = new_node
            return
        last = self.head
        while last.next:
            last = last.next
        last.next = new_node

    def delete(self, key):
        curr = self.head
        prev = None
        while curr:
            if curr.data == key:
                if prev:
                    prev.next = curr.next
                else:
                    self.head = curr.next
                return
            prev = curr
            curr = curr.next

    def traverse(self):
        elems = []
        curr = self.head
        while curr:
            elems.append(curr.data)
            curr = curr.next
        return elems

class LinkedListOperationsGUI:
    def __init__(self, root):
        self.linked_list = LinkedList()
        self.root = root
        self.root.title("Linked List Operations")
        self.root.geometry("500x400")
        self.create_widgets()

    def create_widgets(self):
        self.element_label = ctk.CTkLabel(self.root, text="Element:", font=font_all)
        self.element_label.pack(pady=(20, 5))

        self.element_entry = ctk.CTkEntry(self.root, font=font_all)
        self.element_entry.pack(pady=(0, 20))

        self.append_button = ctk.CTkButton(self.root, text="Append", command=self.append, font=font_all, border_width=2, border_color="white")
        self.append_button.pack(pady=(5, 10))

        self.delete_button = ctk.CTkButton(self.root, text="Delete", command=self.delete, font=font_all, border_width=2, border_color="white")
        self.delete_button.pack(pady=(5, 10))

        self.traverse_button = ctk.CTkButton(self.root, text="Traverse", command=self.traverse, font=font_all, border_width=2, border_color="white")
        self.traverse_button.pack(pady=(5, 10))

        self.result_label = ctk.CTkLabel(self.root, text="", font=font_all)
        self.result_label.pack(pady=(10, 20))

    def append(self):
        element = self.element_entry.get()
        if element:
            self.linked_list.append(element)
            self.result_label.configure(text=f"Appended: {element}")
            self.element_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Input Error", "Please enter an element to append.")

    def delete(self):
        element = self.element_entry.get()
        if element:
            self.linked_list.delete(element)
            self.result_label.configure(text=f"Deleted: {element}")
            self.element_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Input Error", "Please enter an element to delete.")

    def traverse(self):
        traversal = self.linked_list.traverse()
        self.result_label.configure(text=f"Linked List: {traversal}")


class DoublyLinkedList:
    def __init__(self):
        self.head = None

    class Node:
        def __init__(self, data):
            self.data = data
            self.next = None
            self.prev = None

    def append(self, data):
        new_node = self.Node(data)
        if self.head is None:
            self.head = new_node
            return
        last = self.head
        while last.next:
            last = last.next
        last.next = new_node
        new_node.prev = last

    def delete(self, key):
        curr = self.head
        while curr:
            if curr.data == key:
                if curr.prev:
                    curr.prev.next = curr.next
                if curr.next:
                    curr.next.prev = curr.prev
                if curr == self.head:  # Move head if needed
                    self.head = curr.next
                return
            curr = curr.next

    def traverse(self):
        elems = []
        curr = self.head
        while curr:
            elems.append(curr.data)
            curr = curr.next
        return elems

class DoublyLinkedListOperationsGUI:
    def __init__(self, root):
        self.doubly_linked_list = DoublyLinkedList()
        self.root = root
        self.root.title("Doubly Linked List Operations")
        self.root.geometry("500x400")
        self.create_widgets()

    def create_widgets(self):
        self.element_label = ctk.CTkLabel(self.root, text="Element:", font=font_all)
        self.element_label.pack(pady=(20, 5))

        self.element_entry = ctk.CTkEntry(self.root, font=font_all)
        self.element_entry.pack(pady=(0, 20))

        self.append_button = ctk.CTkButton(self.root, text="Append", command=self.append, font=font_all, border_width=2, border_color="white")
        self.append_button.pack(pady=(5, 10))

        self.delete_button = ctk.CTkButton(self.root, text="Delete", command=self.delete, font=font_all, border_width=2, border_color="white")
        self.delete_button.pack(pady=(5, 10))

        self.traverse_button = ctk.CTkButton(self.root, text="Traverse", command=self.traverse, font=font_all, border_width=2, border_color="white")
        self.traverse_button.pack(pady=(5, 10))

        self.result_label = ctk.CTkLabel(self.root, text="", font=font_all)
        self.result_label.pack(pady=(10, 20))

    def append(self):
        element = self.element_entry.get()
        if element:
            self.doubly_linked_list.append(element)
            self.result_label.configure(text=f"Appended: {element}")
            self.element_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Input Error", "Please enter an element to append.")

    def delete(self):
        element = self.element_entry.get()
        if element:
            self.doubly_linked_list.delete(element)
            self.result_label.configure(text=f"Deleted: {element}")
            self.element_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Input Error", "Please enter an element to delete.")

    def traverse(self):
        traversal = self.doubly_linked_list.traverse()
        self.result_label.configure(text=f"Doubly Linked List: {traversal}")

class PriorityQueue:
    def __init__(self):
        self.items = []

    def enqueue(self, item, priority):
        self.items.append((item, priority))
        self.items.sort(key=lambda x: x[1])

    def dequeue(self):
        return self.items.pop(0)[0] if not self.is_empty() else "Priority Queue is empty"

    def is_empty(self):
        return len(self.items) == 0

    def traverse(self):
        return [f"Item: {item}, Priority: {priority}" for item, priority in self.items] if not self.is_empty() else "Priority Queue is empty"

class PriorityQueueOperationsGUI:
    def __init__(self, root):
        self.queue = PriorityQueue()
        self.root = root
        self.root.title("Priority Queue Operations")
        self.root.geometry("500x400")
        self.create_widgets()

    def create_widgets(self):
        self.element_label = ctk.CTkLabel(self.root, text="Element:", font=font_all)
        self.element_label.pack(pady=(20, 5))

        self.element_entry = ctk.CTkEntry(self.root, font=font_all)
        self.element_entry.pack(pady=(0, 20))

        self.priority_label = ctk.CTkLabel(self.root, text="Priority:", font=font_all)
        self.priority_label.pack(pady=(20, 5))

        self.priority_entry = ctk.CTkEntry(self.root, font=font_all)
        self.priority_entry.pack(pady=(0, 20))

        self.enqueue_button = ctk.CTkButton(self.root, text="Enqueue", command=self.enqueue, font=font_all, border_width=2, border_color="white")
        self.enqueue_button.pack(pady=(5, 10))

        self.dequeue_button = ctk.CTkButton(self.root, text="Dequeue", command=self.dequeue, font=font_all, border_width=2, border_color="white")
        self.dequeue_button.pack(pady=(5, 10))

        self.traverse_button = ctk.CTkButton(self.root, text="Traverse", command=self.traverse, font=font_all, border_width=2, border_color="white")
        self.traverse_button.pack(pady=(5, 10))

        self.result_label = ctk.CTkLabel(self.root, text="", font=font_all)
        self.result_label.pack(pady=(10, 20))

    def enqueue(self):
        element = self.element_entry.get()
        priority = self.priority_entry.get()
        if element and priority.isdigit():
            self.queue.enqueue(element, int(priority))
            self.result_label.configure(text=f"Enqueued: {element} with priority {priority}")
            self.element_entry.delete(0, tk.END)
            self.priority_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Input Error", "Please enter valid element and priority.")

    def dequeue(self):
        dequeued = self.queue.dequeue()
        self.result_label.configure(text=f"Dequeued: {dequeued}")

    def traverse(self):
        traversal = self.queue.traverse()
        self.result_label.configure(text=f"Priority Queue: {traversal}")

class BinaryTree:
    class Node:
        def __init__(self, key):
            self.left = None
            self.right = None
            self.val = key

    def __init__(self):
        self.root = None

    def insert(self, key):
        if self.root is None:
            self.root = self.Node(key)
        else:
            self._insert_rec(self.root, key)

    def _insert_rec(self, root, key):
        if key < root.val:
            if root.left is None:
                root.left = self.Node(key)
            else:
                self._insert_rec(root.left, key)
        else:
            if root.right is None:
                root.right = self.Node(key)
            else:
                self._insert_rec(root.right, key)

    def traverse(self):
        return self._in_order_traversal(self.root)

    def _in_order_traversal(self, node):
        if node is not None:
            return self._in_order_traversal(node.left) + [node.val] + self._in_order_traversal(node.right)
        return []

    def visualize(self):
        G = nx.Graph()
        def add_edges(node):
            if node is None:
                return
            if node.left:
                G.add_edge(node.val, node.left.val)
                add_edges(node.left)
            if node.right:
                G.add_edge(node.val, node.right.val)
                add_edges(node.right)
        
        add_edges(self.root)
        nx.draw(G, with_labels=True, node_color='lightblue', node_size=2000, font_size=16, font_color='black')
        plt.title("Binary Tree Visualization")
        plt.show()

class BinaryTreeOperationsGUI:
    def __init__(self, root):
        self.binary_tree = BinaryTree()
        self.root = root
        self.root.title("Binary Tree Operations")
        self.root.geometry("500x400")
        self.create_widgets()

    def create_widgets(self):
        self.element_label = ctk.CTkLabel(self.root, text="Element:", font=font_all)
        self.element_label.pack(pady=(20, 5))

        self.element_entry = ctk.CTkEntry(self.root, font=font_all)
        self.element_entry.pack(pady=(0, 20))

        self.insert_button = ctk.CTkButton(self.root, text="Insert", command=self.insert, font=font_all, border_width=2, border_color="white")
        self.insert_button.pack(pady=(5, 10))

        self.traverse_button = ctk.CTkButton(self.root, text="Traverse", command=self.traverse, font=font_all, border_width=2, border_color="white")
        self.traverse_button.pack(pady=(5, 10))

        self.visualize_button = ctk.CTkButton(self.root, text="Visualize Tree", command=self.visualize_tree, font=font_all, border_width=2, border_color="white")
        self.visualize_button.pack(pady=(5, 10))

        self.result_label = ctk.CTkLabel(self.root, text="", font=font_all)
        self.result_label.pack(pady=(10, 20))

    def insert(self):
        element = self.element_entry.get()
        if element.isdigit():
            self.binary_tree.insert(int(element))
            self.result_label.configure(text=f"Inserted: {element}")
            self.element_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Input Error", "Please enter a valid integer.")

    def traverse(self):
        traversal = self.binary_tree.traverse()
        self.result_label.configure(text=f"Binary Tree (In-Order): {traversal}")

    def visualize_tree(self):
        self.binary_tree.visualize()

class HuffmanCoding:
    class Node:
        def __init__(self, char, freq):
            self.char = char
            self.freq = freq
            self.left = None
            self.right = None

    def __init__(self):
        self.root = None

    def build_tree(self, chars, freqs):
        queue = []
        for char, freq in zip(chars, freqs):
            queue.append(self.Node(char, freq))
        queue.sort(key=lambda x: x.freq)

        while len(queue) > 1:
            left = queue.pop(0)
            right = queue.pop(0)
            merged = self.Node(None, left.freq + right.freq)
            merged.left = left
            merged.right = right
            queue.append(merged)
            queue.sort(key=lambda x: x.freq)

        self.root = queue[0]

    def get_codes(self):
        codes = {}
        self._get_codes_helper(self.root, "", codes)
        return codes

    def _get_codes_helper(self, node, current_code, codes):
        if node is None:
            return
        if node.char is not None:
            codes[node.char] = current_code
        self._get_codes_helper(node.left, current_code + "0", codes)
        self._get_codes_helper(node.right, current_code + "1", codes)

class HuffmanCodingOperationsGUI:
    def __init__(self, root):
        self.huffman_coding = HuffmanCoding()
        self.root = root
        self.root.title("Huffman Coding Operations")
        self.root.geometry("500x400")
        self.create_widgets()

    def create_widgets(self):
        self.char_label = ctk.CTkLabel(self.root, text="Characters (comma-separated):", font=font_all)
        self.char_label.pack(pady=(20, 5))

        self.char_entry = ctk.CTkEntry(self.root, font=font_all)
        self.char_entry.pack(pady=(0, 20))

        self.freq_label = ctk.CTkLabel(self.root, text="Frequencies (comma-separated):", font=font_all)
        self.freq_label.pack(pady=(20, 5))

        self.freq_entry = ctk.CTkEntry(self.root, font=font_all)
        self.freq_entry.pack(pady=(0, 20))

        self.build_button = ctk.CTkButton(self.root, text="Build Tree", command=self.build_tree, font=font_all, border_width=2, border_color="white")
        self.build_button.pack(pady=(5, 10))

        self.codes_button = ctk.CTkButton(self.root, text="Get Codes", command=self.get_codes, font=font_all, border_width=2, border_color="white")
        self.codes_button.pack(pady=(5, 10))

        self.result_label = ctk.CTkLabel(self.root, text="", font=font_all)
        self.result_label.pack(pady=(10, 20))

    def build_tree(self):
        chars = self.char_entry.get().split(',')
        freqs = self.freq_entry.get().split(',')
        if len(chars) == len(freqs) and all(freq.isdigit() for freq in freqs):
            freqs = list(map(int, freqs))
            self.huffman_coding.build_tree(chars, freqs)
            self.result_label.configure(text="Huffman Tree Built!")
            self.char_entry.delete(0, tk.END)
            self.freq_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Input Error", "Please enter valid characters and frequencies.")

    def get_codes(self):
        codes = self.huffman_coding.get_codes()
        self.result_label.configure(text=f"Huffman Codes: {codes}")

class GraphDFS:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)

    def dfs(self, start):
        visited = set()
        result = []
        self._dfs_helper(start, visited, result)
        return result

    def _dfs_helper(self, node, visited, result):
        if node not in visited:
            visited.add(node)
            result.append(node)
            for neighbor in self.graph.get(node, []):
                self._dfs_helper(neighbor, visited, result)

    def visualize(self):
        G = nx.Graph()
        for node, edges in self.graph.items():
            for neighbor in edges:
                G.add_edge(node, neighbor)
        nx.draw(G, with_labels=True, node_color='skyblue', node_size=2000, font_size=16, font_color='black')
        plt.title("Graph Visualization")
        plt.show()

class GraphDFSOperationsGUI:
    def __init__(self, root):
        self.graph_dfs = GraphDFS()
        self.root = root
        self.root.title("Graph DFS Operations")
        self.root.geometry("500x400")
        self.create_widgets()

    def create_widgets(self):
        self.node_label = ctk.CTkLabel(self.root, text="Node (u):", font=font_all)
        self.node_label.pack(pady=(20, 5))

        self.node_entry = ctk.CTkEntry(self.root, font=font_all)
        self.node_entry.pack(pady=(0, 20))

        self.edge_label = ctk.CTkLabel(self.root, text="Connected Node (v):", font=font_all)
        self.edge_label.pack(pady=(20, 5))

        self.edge_entry = ctk.CTkEntry(self.root, font=font_all)
        self.edge_entry.pack(pady=(0, 20))

        self.add_button = ctk.CTkButton(self.root, text="Add Edge", command=self.add_edge, font=font_all, border_width=2, border_color="white")
        self.add_button.pack(pady=(5, 10))

        self.start_label = ctk.CTkLabel(self.root, text="Start Node for DFS:", font=font_all)
        self.start_label.pack(pady=(20, 5))

        self.start_entry = ctk.CTkEntry(self.root, font=font_all)
        self.start_entry.pack(pady=(0, 20))

        self.dfs_button = ctk.CTkButton(self.root, text="Perform DFS", command=self.perform_dfs, font=font_all, border_width=2, border_color="white")
        self.dfs_button.pack(pady=(5, 10))

        self.visualize_button = ctk.CTkButton(self.root, text="Visualize Graph", command=self.visualize_graph, font=font_all, border_width=2, border_color="white")
        self.visualize_button.pack(pady=(5, 10))

        self.result_label = ctk.CTkLabel(self.root, text="", font=font_all)
        self.result_label.pack(pady=(10, 20))

    def add_edge(self):
        u = self.node_entry.get()
        v = self.edge_entry.get()
        if u and v:
            self.graph_dfs.add_edge(u, v)
            self.result_label.configure(text=f"Edge added: {u} -> {v}")
            self.node_entry.delete(0, tk.END)
            self.edge_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Input Error", "Please enter both nodes.")

    def perform_dfs(self):
        start = self.start_entry.get()
        if start:
            traversal = self.graph_dfs.dfs(start)
            self.result_label.configure(text=f"DFS Traversal: {traversal}")
            self.start_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Input Error", "Please enter a starting node.")

    def visualize_graph(self):
        self.graph_dfs.visualize()

class GraphBFS:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)

    def bfs(self, start):
        visited = set()
        queue = [start]
        visited.add(start)
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return result

    def visualize(self):
        G = nx.Graph()
        for node, edges in self.graph.items():
            for neighbor in edges:
                G.add_edge(node, neighbor)
        nx.draw(G, with_labels=True, node_color='lightgreen', node_size=2000, font_size=16, font_color='black')
        plt.title("Graph Visualization")
        plt.show()

class GraphBFSOperationsGUI:
    def __init__(self, root):
        self.graph_bfs = GraphBFS()
        self.root = root
        self.root.title("Graph BFS Operations")
        self.root.geometry("500x400")
        self.create_widgets()

    def create_widgets(self):
        self.node_label = ctk.CTkLabel(self.root, text="Node (u):", font=font_all)
        self.node_label.pack(pady=(20, 5))

        self.node_entry = ctk.CTkEntry(self.root, font=font_all)
        self.node_entry.pack(pady=(0, 20))

        self.edge_label = ctk.CTkLabel(self.root, text="Connected Node (v):", font=font_all)
        self.edge_label.pack(pady=(20, 5))

        self.edge_entry = ctk.CTkEntry(self.root, font=font_all)
        self.edge_entry.pack(pady=(0, 20))

        self.add_button = ctk.CTkButton(self.root, text="Add Edge", command=self.add_edge, font=font_all, border_width=2, border_color="white")
        self.add_button.pack(pady=(5, 10))

        self.start_label = ctk.CTkLabel(self.root, text="Start Node for BFS:", font=font_all)
        self.start_label.pack(pady=(20, 5))

        self.start_entry = ctk.CTkEntry(self.root, font=font_all)
        self.start_entry.pack(pady=(0, 20))

        self.bfs_button = ctk.CTkButton(self.root, text="Perform BFS", command=self.perform_bfs, font=font_all, border_width=2, border_color="white")
        self.bfs_button.pack(pady=(5, 10))

        self.visualize_button = ctk.CTkButton(self.root, text="Visualize Graph", command=self.visualize_graph, font=font_all, border_width=2, border_color="white")
        self.visualize_button.pack(pady=(5, 10))

        self.result_label = ctk.CTkLabel(self.root, text="", font=font_all)
        self.result_label.pack(pady=(10, 20))

    def add_edge(self):
        u = self.node_entry.get()
        v = self.edge_entry.get()
        if u and v:
            self.graph_bfs.add_edge(u, v)
            self.result_label.configure(text=f"Edge added: {u} -> {v}")
            self.node_entry.delete(0, tk.END)
            self.edge_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Input Error", "Please enter both nodes.")

    def perform_bfs(self):
        start = self.start_entry.get()
        if start:
            traversal = self.graph_bfs.bfs(start)
            self.result_label.configure(text=f"BFS Traversal: {traversal}")
            self.start_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Input Error", "Please enter a starting node.")

    def visualize_graph(self):
        self.graph_bfs.visualize()

class TSP:
    def __init__(self, graph):
        self.graph = graph

    def tsp(self, start):
        n = len(self.graph)
        visited = [False] * n
        path = []
        min_cost = float('inf')
        self._tsp_helper(start, visited, path, 0, 0, min_cost)

    def _tsp_helper(self, current, visited, path, count, cost, min_cost):
        visited[current] = True
        path.append(current)

        if count == len(self.graph) - 1:
            # Return to start
            cost += self.graph[current][0]
            if cost < min_cost:
                min_cost = cost
            return

        for neighbor in range(len(self.graph)):
            if not visited[neighbor]:
                self._tsp_helper(neighbor, visited, path, count + 1, cost + self.graph[current][neighbor], min_cost)

        path.pop()
        visited[current] = False

class TSPOperationsGUI:
    def __init__(self, root):
        self.graph = []
        self.root = root
        self.root.title("TSP Operations")
        self.root.geometry("500x400")
        self.create_widgets()

    def create_widgets(self):
        self.node_label = ctk.CTkLabel(self.root, text="Graph (comma-separated row of costs):", font=font_all)
        self.node_label.pack(pady=(20, 5))

        self.node_entry = ctk.CTkEntry(self.root, font=font_all)
        self.node_entry.pack(pady=(0, 20))

        self.start_label = ctk.CTkLabel(self.root, text="Start Node:", font=font_all)
        self.start_label.pack(pady=(20, 5))

        self.start_entry = ctk.CTkEntry(self.root, font=font_all)
        self.start_entry.pack(pady=(0, 20))

        self.tsp_button = ctk.CTkButton(self.root, text="Calculate TSP", command=self.calculate_tsp, font=font_all, border_width=2, border_color="white")
        self.tsp_button.pack(pady=(5, 10))

        self.result_label = ctk.CTkLabel(self.root, text="", font=font_all)
        self.result_label.pack(pady=(10, 20))

    def calculate_tsp(self):
        costs = self.node_entry.get().split(',')
        self.graph = [list(map(int, cost.split())) for cost in costs]
        start = int(self.start_entry.get())
        tsp_solver = TSP(self.graph)
        tsp_solver.tsp(start)
        self.result_label.configure(text="TSP calculation complete!")

class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(size)]

    def hash_function(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        index = self.hash_function(key)
        self.table[index].append((key, value))

    def search(self, key):
        index = self.hash_function(key)
        for kv in self.table[index]:
            if kv[0] == key:
                return kv[1]
        return "Not found"

    def delete(self, key):
        index = self.hash_function(key)
        for i, kv in enumerate(self.table[index]):
            if kv[0] == key:
                del self.table[index][i]
                return f"Deleted: {key}"
        return "Key not found"

class HashTableOperationsGUI:
    def __init__(self, root):
        self.hash_table = HashTable(10)
        self.root = root
        self.root.title("Hash Table Operations")
        self.root.geometry("500x400")
        self.create_widgets()

    def create_widgets(self):
        self.key_label = ctk.CTkLabel(self.root, text="Key:", font=font_all)
        self.key_label.pack(pady=(20, 5))

        self.key_entry = ctk.CTkEntry(self.root, font=font_all)
        self.key_entry.pack(pady=(0, 20))

        self.value_label = ctk.CTkLabel(self.root, text="Value:", font=font_all)
        self.value_label.pack(pady=(20, 5))

        self.value_entry = ctk.CTkEntry(self.root, font=font_all)
        self.value_entry.pack(pady=(0, 20))

        self.insert_button = ctk.CTkButton(self.root, text="Insert", command=self.insert, font=font_all, border_width=2, border_color="white")
        self.insert_button.pack(pady=(5, 10))

        self.search_button = ctk.CTkButton(self.root, text="Search", command=self.search, font=font_all, border_width=2, border_color="white")
        self.search_button.pack(pady=(5, 10))

        self.delete_button = ctk.CTkButton(self.root, text="Delete", command=self.delete, font=font_all, border_width=2, border_color="white")
        self.delete_button.pack(pady=(5, 10))

        self.result_label = ctk.CTkLabel(self.root, text="", font=font_all)
        self.result_label.pack(pady=(10, 20))

    def insert(self):
        key = self.key_entry.get()
        value = self.value_entry.get()
        if key and value:
            self.hash_table.insert(key, value)
            self.result_label.configure(text=f"Inserted: {key} -> {value}")
            self.key_entry.delete(0, tk.END)
            self.value_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Input Error", "Please enter both key and value.")

    def search(self):
        key = self.key_entry.get()
        if key:
            value = self.hash_table.search(key)
            self.result_label.configure(text=f"Search Result: {value}")
        else:
            messagebox.showwarning("Input Error", "Please enter a key.")

    def delete(self):
        key = self.key_entry.get()
        if key:
            result = self.hash_table.delete(key)
            self.result_label.configure(text=result)
        else:
            messagebox.showwarning("Input Error", "Please enter a key.")

class HashTableChaining:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(size)]

    def hash_function(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        index = self.hash_function(key)
        for kv in self.table[index]:
            if kv[0] == key:
                kv[1] = value
                return
        self.table[index].append([key, value])

    def search(self, key):
        index = self.hash_function(key)
        for kv in self.table[index]:
            if kv[0] == key:
                return kv[1]
        return "Not found"

    def delete(self, key):
        index = self.hash_function(key)
        for i, kv in enumerate(self.table[index]):
            if kv[0] == key:
                del self.table[index][i]
                return f"Deleted: {key}"
        return "Key not found"

class HashTableChainingOperationsGUI:
    def __init__(self, root):
        self.hash_table = HashTableChaining(10)
        self.root = root
        self.root.title("Hash Table Chaining Operations")
        self.root.geometry("500x400")
        self.create_widgets()

    def create_widgets(self):
        self.key_label = ctk.CTkLabel(self.root, text="Key:", font=font_all)
        self.key_label.pack(pady=(20, 5))

        self.key_entry = ctk.CTkEntry(self.root, font=font_all)
        self.key_entry.pack(pady=(0, 20))

        self.value_label = ctk.CTkLabel(self.root, text="Value:", font=font_all)
        self.value_label.pack(pady=(20, 5))

        self.value_entry = ctk.CTkEntry(self.root, font=font_all)
        self.value_entry.pack(pady=(0, 20))

        self.insert_button = ctk.CTkButton(self.root, text="Insert", command=self.insert, font=font_all, border_width=2, border_color="white")
        self.insert_button.pack(pady=(5, 10))

        self.search_button = ctk.CTkButton(self.root, text="Search", command=self.search, font=font_all, border_width=2, border_color="white")
        self.search_button.pack(pady=(5, 10))

        self.delete_button = ctk.CTkButton(self.root, text="Delete", command=self.delete, font=font_all, border_width=2, border_color="white")
        self.delete_button.pack(pady=(5, 10))

        self.result_label = ctk.CTkLabel(self.root, text="", font=font_all)
        self.result_label.pack(pady=(10, 20))

    def insert(self):
        key = self.key_entry.get()
        value = self.value_entry.get()
        if key and value:
            self.hash_table.insert(key, value)
            self.result_label.configure(text=f"Inserted: {key} -> {value}")
            self.key_entry.delete(0, tk.END)
            self.value_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Input Error", "Please enter both key and value.")

    def search(self):
        key = self.key_entry.get()
        if key:
            value = self.hash_table.search(key)
            self.result_label.configure(text=f"Search Result: {value}")
        else:
            messagebox.showwarning("Input Error", "Please enter a key.")

    def delete(self):
        key = self.key_entry.get()
        if key:
            result = self.hash_table.delete(key)
            self.result_label.configure(text=result)
        else:
            messagebox.showwarning("Input Error", "Please enter a key.")

class MainMenu:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Structure Operations Dashboard")
        self.root.geometry("600x500")
        self.create_widgets()

    def create_widgets(self):
        self.label = ctk.CTkLabel(self.root, text="Select an Operation", font=("Work Sans", 24))
        self.label.pack(pady=20)

        self.queue_button = ctk.CTkButton(self.root, text="Queue Operations", command=self.open_queue_gui, font=font_all, border_width=2, border_color="white")
        self.queue_button.pack(pady=10)

        self.stack_button = ctk.CTkButton(self.root, text="Stack Operations", command=self.open_stack_gui, font=font_all, border_width=2, border_color="white")
        self.stack_button.pack(pady=10)

        self.linked_list_button = ctk.CTkButton(self.root, text="Linked List Operations", command=self.open_linked_list_gui, font=font_all, border_width=2, border_color="white")
        self.linked_list_button.pack(pady=10)

        self.doubly_linked_list_button = ctk.CTkButton(self.root, text="Doubly Linked List Operations", command=self.open_doubly_linked_list_gui, font=font_all, border_width=2, border_color="white")
        self.doubly_linked_list_button.pack(pady=10)

        self.priority_queue_button = ctk.CTkButton(self.root, text="Priority Queue Operations", command=self.open_priority_queue_gui, font=font_all, border_width=2, border_color="white")
        self.priority_queue_button.pack(pady=10)

        self.binary_tree_button = ctk.CTkButton(self.root, text="Binary Tree Operations", command=self.open_binary_tree_gui, font=font_all, border_width=2, border_color="white")
        self.binary_tree_button.pack(pady=10)

        self.huffman_coding_button = ctk.CTkButton(self.root, text="Huffman Coding Operations", command=self.open_huffman_coding_gui, font=font_all, border_width=2, border_color="white")
        self.huffman_coding_button.pack(pady=10)

        self.graph_dfs_button = ctk.CTkButton(self.root, text="Graph DFS Operations", command=self.open_graph_dfs_gui, font=font_all, border_width=2, border_color="white")
        self.graph_dfs_button.pack(pady=10)

        self.graph_bfs_button = ctk.CTkButton(self.root, text="Graph BFS Operations", command=self.open_graph_bfs_gui, font=font_all, border_width=2, border_color="white")
        self.graph_bfs_button.pack(pady=10)

        self.tsp_button = ctk.CTkButton(self.root, text="TSP Operations", command=self.open_tsp_gui, font=font_all, border_width=2, border_color="white")
        self.tsp_button.pack(pady=10)

        self.hash_table_button = ctk.CTkButton(self.root, text="Hash Table Operations", command=self.open_hash_table_gui, font=font_all, border_width=2, border_color="white")
        self.hash_table_button.pack(pady=10)

        self.hash_table_chaining_button = ctk.CTkButton(self.root, text="Hash Table Chaining Operations", command=self.open_hash_table_chaining_gui, font=font_all, border_width=2, border_color="white")
        self.hash_table_chaining_button.pack(pady=10)

    def open_queue_gui(self):
        queue_window = ctk.CTkToplevel(self.root)
        queue_window.geometry("500x400")
        app = QueueOperationsGUI(queue_window)

    def open_stack_gui(self):
        stack_window = ctk.CTkToplevel(self.root)
        stack_window.geometry("500x400")
        app = StackOperationsGUI(stack_window)

    def open_linked_list_gui(self):
        linked_list_window = ctk.CTkToplevel(self.root)
        linked_list_window.geometry("500x400")
        app = LinkedListOperationsGUI(linked_list_window)

    def open_doubly_linked_list_gui(self):
        doubly_linked_list_window = ctk.CTkToplevel(self.root)
        doubly_linked_list_window.geometry("500x400")
        app = DoublyLinkedListOperationsGUI(doubly_linked_list_window)

    def open_priority_queue_gui(self):
        priority_queue_window = ctk.CTkToplevel(self.root)
        priority_queue_window.geometry("500x400")
        app = PriorityQueueOperationsGUI(priority_queue_window)

    def open_binary_tree_gui(self):
        binary_tree_window = ctk.CTkToplevel(self.root)
        binary_tree_window.geometry("500x400")
        app = BinaryTreeOperationsGUI(binary_tree_window)

    def open_huffman_coding_gui(self):
        huffman_window = ctk.CTkToplevel(self.root)
        huffman_window.geometry("500x400")
        app = HuffmanCodingOperationsGUI(huffman_window)

    def open_graph_dfs_gui(self):
        graph_dfs_window = ctk.CTkToplevel(self.root)
        graph_dfs_window.geometry("500x400")
        app = GraphDFSOperationsGUI(graph_dfs_window)

    def open_graph_bfs_gui(self):
        graph_bfs_window = ctk.CTkToplevel(self.root)
        graph_bfs_window.geometry("500x400")
        app = GraphBFSOperationsGUI(graph_bfs_window)

    def open_tsp_gui(self):
        tsp_window = ctk.CTkToplevel(self.root)
        tsp_window.geometry("500x400")
        app = TSPOperationsGUI(tsp_window)

    def open_hash_table_gui(self):
        hash_table_window = ctk.CTkToplevel(self.root)
        hash_table_window.geometry("500x400")
        app = HashTableOperationsGUI(hash_table_window)

    def open_hash_table_chaining_gui(self):
        hash_table_chaining_window = ctk.CTkToplevel(self.root)
        hash_table_chaining_window.geometry("500x400")
        app = HashTableChainingOperationsGUI(hash_table_chaining_window)

root = ctk.CTk()
app = MainMenu(root)
root.mainloop()