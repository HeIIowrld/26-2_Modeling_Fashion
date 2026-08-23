"""유효한 무신사 카테고리 코드를 찾는다. 다양성을 넓히려면 카테고리를 늘리는 게
페이지를 깊게 파는 것보다 낫다(깊이 파면 인기 없는 상품만 더 나온다)."""
import sys, time
sys.path.insert(0, "/data1/dsl01/26-2_Modeling_Fashion/ai_fashion_recommender/scripts")
import musinsa_crawler as mc

# 상의(001)·아우터(002)·바지(003)·스커트(100) 하위 코드를 훑는다
CANDIDATES = []
for base in ("001", "002", "003"):
    CANDIDATES += [f"{base}{i:03d}" for i in range(1, 26)]
CANDIDATES += ["100", "020", "022"]

good = {}
for code in CANDIDATES:
    try:
        items = mc.fetch_page(code, 1, "POPULAR", 3)
        if items:
            good[code] = (items[0].get("goodsName") or "")[:34]
    except Exception:
        pass
    time.sleep(0.35)

print(f"유효 카테고리 {len(good)}개\n")
for code, sample in sorted(good.items()):
    print(f"  {code}  {sample}")
