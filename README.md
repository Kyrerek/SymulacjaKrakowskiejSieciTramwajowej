# Symulacja krakowskiej sieci tramwajowej z opóźnieniami
## Dane
Dane GTFS Realtime są zbierane od 18.10.2026. Trip updates ma przykładowy plik, ponieważ dane z całego dnia przekraczają limit GitHuba 🙂
## Model
Wykorzystamy model grafowy - Descrit Event Simulation. Węzłami grafu będą przystanki oraz np. sygnalizacja i przejścia dla pieszych, żeby dokładniej zasymulować jazde po mieście. Ewentualnie można spróbować zasymulować przeszkody (sygnalizajce itp.) za pomocą losowych opóźnień na połączeniach węzłów. 
Minusem jest mniej efektowna wizualizacja, ale ten model powinien być szybszy do zrobienia oraz w działaniu.

Ewentualnością jest model komórkowy, który jest raczej trudniejszy do implementacji i będzie wolniej działał przez złożoność obliczeniową. 
Jednak da to lepszą wizualizacje, model ruchu pojazdów (hamowanie i wizualne kolejki przed przystankami) i pozwoli nadawać komórkom właściwości, np. bycie sygnalizacją (problem: dane o cyklu świateł).
Ogólny zarys: cała sieć tramwajowa jest złożona z komórek, po których poruszają się tramwaje z jakąś prędkością (n komórek na sekunde).