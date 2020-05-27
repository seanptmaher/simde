(defpackage testgen
  (:use :cl :cl-user)
  (:import-from :cl-interpol)
  (:export :test-gen))
(in-package testgen)

(named-readtables:in-readtable :interpol-syntax)

(defun test-gen (test-name test-func num-tests
                 vec-type vec-load-func vec-elem-load-func vec-len
                 vec-rand-gen-func vec-trans-func assert-func)
  (let (test-values testcases)
    (dotimes (n num-tests)
      (let (curr)
        (push '() curr) (push '() curr) (push '() curr)
        (dotimes (m vec-len)
          (push (funcall vec-rand-gen-func) (first curr))
          (push (funcall vec-rand-gen-func) (second curr)))
        (setf (third curr) (funcall vec-trans-func (first curr) (second curr)))
        (push curr test-values)))

    (setf testcases (apply #'concatenate 'string
                            (let (acc)
                              (dolist (tc test-values)
                                (push
                                 (apply #'concatenate  'string
                                        (append (cons #?|{ | nil)
                                                (let (acc2)
                                                  (dotimes (i 3)
                                                    (when (not (= i 0)) (push "
      " acc2))
                                                    (push #?|${vec-load-func}(| acc2)
                                                    (dotimes (j vec-len)
                                                      (push #?| ${vec-elem-load-func}(${(nth j (nth i tc))}), |
                                                            acc2)))
                                                  (push " }," acc2)
                                                  (push (make-string 1 :initial-element #\Newline) acc2)
                                                  (push "    " acc2)
                                                  (nreverse acc2))))
                                 acc))
                              acc)))
    #?|static MunitResult
${test-name}(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    ${vec-type} a;
    ${vec-type} b;
    ${vec-type} r;
  } test_vec[${num-tests}] = {
    ${testcases}};

  for (size_t i = 0; i < (sizeof(test_vec) / sizeof(test_vec[0])); ++i) {
    ${vec-type} r = ${test-func}(test_vec[i].a, test_vec[i].b);
    ${assert-func}(r, ==, test_vec[i].r);    
  }

  return MUNIT_OK;
}|))

