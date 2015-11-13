
    if norm( x - x_old, 'inf' ) / max( 1, norm( x_old, 'inf' ) ) <= xtol 
      mflag    = FLAG_XTOL;
      message = MESSAGE_XTOL;
      loop    = 0;
    elseif abs( f_old - f_x ) / max( 1, abs( f_old ) ) <= ftol
      mflag    = FLAG_FTOL;
      message = MESSAGE_FTOL;
      loop    = 0;
    elseif iter >= max_iter 
      mflag    = FLAG_MAXITER;
      message = MESSAGE_MAXITER;
      loop    = 0;
    elseif fun_evals >= max_fun_evals
      mflag    = FLAG_MAXFUNEV;
      message = MESSAGE_MAXFUNEV;
      loop    = 0;
    end
