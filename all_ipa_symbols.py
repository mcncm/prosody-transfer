stress_filename = 'tacotron2/ipa-filelists/ljs_audio_text_test_filelist.txt'
nostress_filename = 'tacotron2/nostress-ipa-filelists/ljs_audio_text_test_filelist.txt'
stress_out_file = 'stress_ipa_symbols.txt'
nostress_out_file = 'nostress_ipa_symbols.txt'

# stressed characters
with open(stress_filename, 'r') as f:
    charset = {c for line in f.readlines() for c in line.split('|')[1]
               if c not in [' ', '\n']}
    chars = ''.join(list(charset))

with open(stress_out_file, 'w') as g:
    g.write(chars)


# unstressed characters
with open(nostress_filename, 'r') as f:
    charset = {c for line in f.readlines() for c in line.split('|')[1]
               if c not in [' ', '\n']}
    chars = ''.join(list(charset))

with open(nostress_out_file, 'w') as g:
    g.write(chars)
