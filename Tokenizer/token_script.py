from regextokenizer import RegexTokenizer

# training file
with open("Tokenizer/taylorswift.txt", "r") as f:
    text = f.read()
# call tokenider
tk = RegexTokenizer()
print(len(tk.vocab_dict))
print(len(tk.merge_dict))
vocab_size = 256 + 100
# tk.train(text, vocab_size, verbose=True)
tk.check_train(verbose=True)
print(f"merge = {len(tk.merge_dict)}")
print(f"vocab= {len(tk.vocab_dict)}")
print(tk.decode(tk.encode("Hello, my name is Vicente, how are you?")))
# print(tk.decode([32, 234, 2, 1, 3, 423, 2, 43, 342, 324, 2342, 133]))

# 3st0YunP0C04rt0**
