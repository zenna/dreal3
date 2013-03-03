exception CException of string

type key = string
type intv = Intv.t
type t = (key, intv) BatPMap.t

let find (x : key) (e : t) : intv
    = BatPMap.find x e

let order (e1 : t) (e2 : t) : bool =
  BatPMap.for_all
    (fun x i1 ->
      let i2 = find x e2 in
      Intv.order i1 i2
    )
    e1

let join (e1 : t) (e2 : t) : t =
  BatPMap.merge
    (fun x i1_op i2_op ->
      match (i1_op, i2_op) with
          (Some i1, Some i2) -> Some (Intv.join i1 i2)
        | _ -> raise (CException "Merge fail"))
    e1
    e2

let print out =
  BatPMap.print ~first:"{" ~last:"}\n" ~sep:", \n"
    BatString.print
    Intv.print
    out

let to_list (e : t) : (key * intv) list
    = BatList.of_enum (BatPMap.backwards e)

let from_list (l : (key * intv) list) : t =
  List.fold_left
    (fun e (k, i) -> BatPMap.add k i e)
    BatPMap.empty
    l

let make = from_list

let equals (e1 : t) (e2 : t) : bool =
  not (List.mem
       false
       (List.map
          (fun ((_, i1), (_, i2)) ->
            Intv.equals i1 i2)
          (BatList.combine (to_list e1) (to_list e1))))

let is_empty (e : t) : bool =
  List.mem true
    (List.map
       (fun (_, {Intv.low = l; Intv.high = h})
       -> (BatFloat.compare l h) = 0)
       (to_list e))

(* minus e1 e2 == (e1 - e2) *)
let minus (e1 : t) (e2 : t) : (t list) =
  let extract_diff_dim l1 l2 =
    let diff_list =
      List.filter
        (fun ((_, i1), (_, i2))
        -> not (Intv.equals i1 i2))
        (BatList.combine l1 l2) in
    match diff_list with
    | hd::[] -> hd
    | _ -> raise (CException ("Two envs differ on multiple dimensions: " ^ string_of_int (List.length diff_list)))
  in
  let l1 = to_list e1 in
  let l2 = to_list e2 in
  let ((key, _), (_, _)) = extract_diff_dim l1 l2 in
  let (l1', l2') =
    BatList.split
      (List.map
         (fun (((key1, {Intv.low = l1; Intv.high = h1}) as elem1),
             ((key2, {Intv.low = l2; Intv.high = h2}) as elem2))
         ->
           if key != key1 then
             (elem1, elem2)
           else
             ((key1, {Intv.low = l1; Intv.high = l2}),
              (key2, {Intv.low = h1; Intv.high = h2}))
         )
         (BatList.combine l1 l2)
      )
  in
  List.filter (fun e -> not (is_empty e)) [from_list l1';from_list l2']