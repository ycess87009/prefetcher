#ifndef TRANSPOSE_IMPL
#define TRANSPOSE_IMPL

#ifdef NAIVE
void transpose(int *src, int *dst, int w, int h)
{
    for (int x = 0; x < w; x++)
        for (int y = 0; y < h; y++)
            *(dst + x * h + y) = *(src + y * w + x);
}
#endif

#ifdef SSE
void transpose(int *src, int *dst, int w, int h)
{
    for (int x = 0; x < w; x += 4) {
        for (int y = 0; y < h; y += 4) {
            __m128i I0 = _mm_loadu_si128((__m128i *)(src + (y + 0) * w + x));
            __m128i I1 = _mm_loadu_si128((__m128i *)(src + (y + 1) * w + x));
            __m128i I2 = _mm_loadu_si128((__m128i *)(src + (y + 2) * w + x));
            __m128i I3 = _mm_loadu_si128((__m128i *)(src + (y + 3) * w + x));
            __m128i T0 = _mm_unpacklo_epi32(I0, I1);
            __m128i T1 = _mm_unpacklo_epi32(I2, I3);
            __m128i T2 = _mm_unpackhi_epi32(I0, I1);
            __m128i T3 = _mm_unpackhi_epi32(I2, I3);
            I0 = _mm_unpacklo_epi64(T0, T1);
            I1 = _mm_unpackhi_epi64(T0, T1);
            I2 = _mm_unpacklo_epi64(T2, T3);
            I3 = _mm_unpackhi_epi64(T2, T3);
            _mm_storeu_si128((__m128i *)(dst + ((x + 0) * h) + y), I0);
            _mm_storeu_si128((__m128i *)(dst + ((x + 1) * h) + y), I1);
            _mm_storeu_si128((__m128i *)(dst + ((x + 2) * h) + y), I2);
            _mm_storeu_si128((__m128i *)(dst + ((x + 3) * h) + y), I3);
        }
    }
}
#endif

#ifdef SSE_PREFETCH

void transpose(int *src, int *dst, int w, int h)
{
    for (int x = 0; x < w; x += 4) {
        for (int y = 0; y < h; y += 4) {
#define PFDIST  8
            _mm_prefetch(src+(y + PFDIST + 0) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 1) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 2) *w + x, _MM_HINT_T1);
            _mm_prefetch(src+(y + PFDIST + 3) *w + x, _MM_HINT_T1);

            __m128i I0 = _mm_loadu_si128 ((__m128i *)(src + (y + 0) * w + x));
            __m128i I1 = _mm_loadu_si128 ((__m128i *)(src + (y + 1) * w + x));
            __m128i I2 = _mm_loadu_si128 ((__m128i *)(src + (y + 2) * w + x));
            __m128i I3 = _mm_loadu_si128 ((__m128i *)(src + (y + 3) * w + x));
            __m128i T0 = _mm_unpacklo_epi32(I0, I1);
            __m128i T1 = _mm_unpacklo_epi32(I2, I3);
            __m128i T2 = _mm_unpackhi_epi32(I0, I1);
            __m128i T3 = _mm_unpackhi_epi32(I2, I3);
            I0 = _mm_unpacklo_epi64(T0, T1);
            I1 = _mm_unpackhi_epi64(T0, T1);
            I2 = _mm_unpacklo_epi64(T2, T3);
            I3 = _mm_unpackhi_epi64(T2, T3);
            _mm_storeu_si128((__m128i *)(dst + ((x + 0) * h) + y), I0);
            _mm_storeu_si128((__m128i *)(dst + ((x + 1) * h) + y), I1);
            _mm_storeu_si128((__m128i *)(dst + ((x + 2) * h) + y), I2);
            _mm_storeu_si128((__m128i *)(dst + ((x + 3) * h) + y), I3);
        }
    }
}
#endif

#ifdef ASM_SSE
void transpose(int *src, int *dst, int w, int h)
{
    for (int x = 0; x < w; x += 4) {
        for (int y = 0; y < h; y += 4) {

            asm volatile
            (
                "movups %[e], %%xmm0 \n\t" //I0
                "movups %[f], %%xmm1 \n\t" //I1
                "movups %[g], %%xmm2 \n\t" //I2
                "movups %[i], %%xmm3 \n\t" //I3

                "movups %%xmm0,%%xmm4 \n\t" //I1
                "movups %%xmm2,%%xmm5 \n\t" //I3

                "punpckldq %%xmm1,%%xmm4 \n\t" //T0  I0 I1
                "punpckldq %%xmm3,%%xmm5 \n\t" //T1  I
                "punpckhdq %%xmm1,%%xmm0 \n\t" //T2
                "punpckhdq %%xmm3,%%xmm2 \n\t" //T3

                "movups %%xmm4,%%xmm1 \n\t"  //T1
                "movups %%xmm0,%%xmm3 \n\t"  //T3

                "punpcklqdq %%xmm5,%%xmm4 \n\t" //I0
                "punpckhqdq %%xmm5,%%xmm1 \n\t" //I1
                "punpcklqdq %%xmm2,%%xmm3 \n\t" //I2
                "punpckhqdq %%xmm2,%%xmm0 \n\t" //I3


                "movups %%xmm4,%[a] \n\t"
                "movups %%xmm1,%[b] \n\t"
                "movups %%xmm3,%[c] \n\t"
                "movups %%xmm0,%[d] \n\t"

                : [a] "=m" ( dst[(x + 0)*h + y] ), [b] "=m" ( dst[(x + 1) * h + y] )
                , [c] "=m" ( dst[(x + 2)*h + y] ), [d]"=m" ( dst[(x + 3) * h + y] )

                : [e] "m" ( src[(y + 0)*w + x] ), [f] "m" ( src[(y + 1) * w + x] )
                , [g] "m" ( src[(y + 2)*w + x] ), [i] "m" ( src[(y + 3) * w + x] )
            );

        }
    }
}
#endif

#ifdef ASM_PREFETCH_SSE
void transpose(int *src, int *dst, int w, int h)
{
    for (int x = 0; x < w; x += 4) {
        for (int y = 0; y < h; y += 4) {
            asm volatile
            (
                "prefetcht1 %[pre1] \n\t"
                "prefetcht1 %[pre2] \n\t"
                "prefetcht1 %[pre3] \n\t"
                "prefetcht1 %[pre4] \n\t"

                "movups %[e], %%xmm0 \n\t" //I0
                "movups %[f], %%xmm1 \n\t" //I1
                "movups %[g], %%xmm2 \n\t" //I2
                "movups %[i], %%xmm3 \n\t" //I3

                "movups %%xmm0,%%xmm4 \n\t" //I1
                "movups %%xmm2,%%xmm5 \n\t" //I3

                "punpckldq %%xmm1,%%xmm4 \n\t" //T0  I0 I1
                "punpckldq %%xmm3,%%xmm5 \n\t" //T1  I
                "punpckhdq %%xmm1,%%xmm0 \n\t" //T2
                "punpckhdq %%xmm3,%%xmm2 \n\t" //T3

                "movups %%xmm4,%%xmm1 \n\t"  //T1
                "movups %%xmm0,%%xmm3 \n\t"  //T3

                "punpcklqdq %%xmm5,%%xmm4 \n\t" //I0
                "punpckhqdq %%xmm5,%%xmm1 \n\t" //I1
                "punpcklqdq %%xmm2,%%xmm3 \n\t" //I2
                "punpckhqdq %%xmm2,%%xmm0 \n\t" //I3


                "movups %%xmm4,%[a] \n\t"
                "movups %%xmm1,%[b] \n\t"
                "movups %%xmm3,%[c] \n\t"
                "movups %%xmm0,%[d] \n\t"

                : [a] "=m" ( dst[(x + 0)*h + y] ), [b] "=m" ( dst[(x + 1) * h + y] )
                , [c] "=m" ( dst[(x + 2)*h + y] ), [d]"=m" ( dst[(x + 3) * h + y] )
                , [pre1] "=m" (src[(y + 8) *w + x]),[pre2] "=m" (src[(y + 9) *w + x])
                , [pre3] "=m" (src[(y + 10) *w + x]),[pre4] "=m" (src[(y + 11) *w + x])

                : [e] "m" ( src[(y + 0)*w + x] ), [f] "m" ( src[(y + 1) * w + x] )
                , [g] "m" ( src[(y + 2)*w + x] ), [i] "m" ( src[(y + 3) * w + x] )
            );

        }
    }
}

#endif

#endif
