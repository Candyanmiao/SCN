function Y = OneHotMatrix(X)
            [N,p] = size(X);
            Y = zeros(N,p);
            if p>1
                for i = 1:N
                    [~,ind] = max(X(i,:));
                    Y(i,ind) = 1;
                end
            else
                for i = 1:N
                    if X(i) > 0.50
                        Y(i) = 1;
                    end
                end
            end
        end 