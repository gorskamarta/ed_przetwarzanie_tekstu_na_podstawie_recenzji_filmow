# ed_przetwarzanie_tekstu_na_podstawie_recenzji_filmow
Design for studies - data mining

# Przetwarzanie tekstu na podstawie recenzji filmów
Dataset: kaggle, IMDB (csv)
Proces przetworzenia:
1. załadowanie pliku CSV
2. usunięcie zbędnych danych wejściowych
3. przebudowa nazw etykiet
4. czyszczenie danych wejściowych
- usunięcie znaczników HTML
- pozostawienie ciągów znaków nie zawierających innych znaków niż litery
- konwersja na małe litery
- usunięcie stopwords
5. Przygotowanie tablic dla ML
6. Wektoryzacja przy pomocy sklearn, CountVectorizer
7. Przygotowanie modelu do nauki. Sprawdzono 3 wersje:
Ilość warstw ukrytych x ilość neuronów
- 3 x 16
- 1 x 16
- 1 x 3
8. Nauka uruchomiona dla 20 epok
9. Przedstawienie wykresów z historii nauki.
