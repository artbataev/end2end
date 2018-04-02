TUR_CHARS = "abcçdefgğhıijklmnoöprsştuüvyz'"

def tur_to_lower(text):
    global TUR_CHARS
    words = text.split()
    words_pure = []
    for word in words:
        if word.casefold() in ["<spoken_noise>", "<silence>", "<hes>", "<unk>", "<v-noise>", "<noise>"]:
            continue
        word = word.replace("\u0130", "i").replace("I", "\u0131").casefold()
        word = word.replace("q", "g").replace("x", "h").replace("\u0430", "a").replace("w", "v")
        pure_word = "".join([c if c in TUR_CHARS else " " for c in word])
        words_pure += pure_word.split()
    return " ".join(words_pure)


def base_to_lower(text, chars):
    words = text.split()
    words_pure = []
    for word in words:
        if word.casefold() in ["<spoken_noise>", "<silence>", "<hes>", "<unk>", "<v-noise>", "<noise>"]:
            continue
        pure_word = "".join([c if c in chars else " " for c in word.casefold()])
        if pure_word and not pure_word.isspace():
            words_pure += pure_word.split()
    return " ".join(words_pure)
