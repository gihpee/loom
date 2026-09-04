/** Телефон ли это.
 *
 *  Ширина, а не разбор userAgent: перечислять устройства — заведомо неполный
 *  список, который устаревает молча. Подписка обязательна: поворот экрана
 *  меняет ответ, и без неё планшет, повёрнутый в портрет, остался бы с тяжёлым
 *  фоном рабочего стола.
 */
import { useEffect, useState } from "react";

const PHONE = "(max-width: 760px)";

export function usePhone() {
  const [phone, setPhone] = useState(
    () => typeof matchMedia === "function" && matchMedia(PHONE).matches,
  );

  useEffect(() => {
    const mq = matchMedia(PHONE);
    const on = () => setPhone(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);

  return phone;
}
