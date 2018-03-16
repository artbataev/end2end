TUR_CHARS = "abcçdefgğhıijklmnoöprsştuüvyz'"

def tur_to_lower(text):
    global TUR_CHARS
    words = text.split()
    words_pure = []
    for word in words:
        if word.lower() in ["<spoken_noise>", "<silence>"]:
            continue
        word = word.replace("\u0130", "i").replace("I", "\u0131").casefold()
        word = word.replace("q", "g").replace("x", "h").replace("\u0430", "a").replace("w", "v")
        pure_word = "".join([c for c in word if c in TUR_CHARS])
        words_pure.append(pure_word)
    return " ".join(words_pure)


def base_to_lower(text, chars):
    words = text.split()
    words_pure = []
    for word in words:
        if word.casefold() in ["<spoken_noise>", "<silence>"]:
            continue
        pure_word = "".join([c for c in word if c in chars])
        words_pure.append(pure_word)
    return " ".join(words_pure)
