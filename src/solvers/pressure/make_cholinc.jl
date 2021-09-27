function make_cholinc(A)
    N = size(A, 1);
    N2 = size(A, 2);

    if N != N2
        error("matrix A should be square");
    end

    [A, d] = spdiagm(A);
    nd = (length(d)+1)/2;

    # take only lower diagonals
    A = A[:, nd:-1:1];
    d = d[nd:end];


    d = [d; N];
    nd = length(d);
    D = zeros(N);

    i=1;
    D[i] = A[i, 1];


    for j=2:nd-1

        for i=d[j]+1:d[j+1]

            s = 0;

            for k=2:j
                s = s - A[i-d[k], k]^2/D[i-d[k]];
            end

            D[i] = A[i, 1] + s;

        end
    end

    D
end
