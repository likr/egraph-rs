import React, { useEffect, useRef } from "react";

export function Wrapper({ onResize, children }) {
  const wrapperRef = useRef();
  useEffect(() => {
    function resize() {
      if (onResize) {
        const { clientWidth, clientHeight } = wrapperRef.current;
        onResize(clientWidth, clientHeight);
      }
    }

    resize();
    window.addEventListener("resize", resize);

    return () => {
      window.removeEventListener("resize", resize);
    };
  }, []);
  return (
    <figure
      ref={wrapperRef}
      className="image is-3by2"
      style={{ boxShadow: "0 0 1em", margin: "1rem" }}
    >
      {children}
    </figure>
  );
}
